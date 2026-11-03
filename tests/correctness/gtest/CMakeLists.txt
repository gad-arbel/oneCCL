# Core sources - no SYCL, no MPI, no plugin
set(CORE_SOURCES
    data/host_buffer.cpp
    data/test_data.cpp
    data/patterns/data_pattern.cpp
    data/patterns/rank_constant_pattern.cpp
    data/orders/sequential_order.cpp
    data/compare/mismatch_report.cpp
    params/collective_params.cpp
    params/param_expander.cpp
    params/validity_table.cpp
    framework/environ_config_backend.cpp
    framework/config_scope.cpp
)

# occl_gtest_data - OBJECT library (not STATIC)
# Forces link errors on dependency violations
add_library(occl_gtest_data OBJECT ${CORE_SOURCES})
# gtest/gtest_main come from deps/googletest (add_subdirectory, not find_package)
target_link_libraries(occl_gtest_data PUBLIC gtest)
target_compile_options(occl_gtest_data PRIVATE "${COVERAGE_CXX_FLAGS}" "${COMMON_CXX_FLAGS}")
target_include_directories(occl_gtest_data PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# occl_test_infra_tests binary - unit tests of the test infrastructure itself, kept apart
# from the suites that test the product
add_executable(occl_test_infra_tests
    suites/test_infra/test_data_test.cpp
    suites/test_infra/config_scope_test.cpp
    suites/test_infra/validity_table_test.cpp
    suites/test_infra/thread_groups_test.cpp
)
target_link_libraries(occl_test_infra_tests
    occl_gtest_data
    gtest_main
    "${COVERAGE_CXX_FLAGS}"
)
target_compile_options(occl_test_infra_tests PRIVATE "${COVERAGE_CXX_FLAGS}" "${COMMON_CXX_FLAGS}")

# Register with ctest
add_test(NAME occl_test_infra_tests
    COMMAND occl_test_infra_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    CONFIGURATIONS unit_tests
)
set_tests_properties(occl_test_infra_tests PROPERTIES
    TIMEOUT 60
)

# ---------------------------------------------------------------------------
# The device half: links oneCCL2, MPI and SYCL. Guarded exactly as the existing
# acceptance targets are, so the core half still compiles where CI turns MPI
# off.
# ---------------------------------------------------------------------------
if(COMPILER_SUPPORTS_SYCL AND ONECCL_TEST_USE_MPI)
    find_package(MPI REQUIRED)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    # occl_gtest_device - the SYCL/MPI/plugin half: the framework files that
    # need MPI, oneCCL or SYCL, plus the fixtures, buffers and executors.
    set(DEVICE_SOURCES
        framework/test_env.cpp
        framework/thread_context.cpp
        framework/test_runtime_environment.cpp
        framework/abort_listener.cpp
        framework/refusal_report.cpp
        fixtures/device_test.cpp
        fixtures/collective_test.cpp
        buffers/device_buffer.cpp
        executors/collective_executor.cpp
        executors/executor_factory.cpp
        executors/allreduce_executor.cpp
    )
    add_library(occl_gtest_device ${DEVICE_SOURCES})
    target_link_libraries(occl_gtest_device PUBLIC
        occl_gtest_data
        oneCCL2
        MPI::MPI_CXX
        Threads::Threads
    )
    target_compile_options(occl_gtest_device PRIVATE "${COVERAGE_CXX_FLAGS}" "${COMMON_CXX_FLAGS}")
    target_link_libraries(occl_gtest_device PRIVATE "${COVERAGE_CXX_FLAGS}")
    target_include_directories(occl_gtest_device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    # SOURCES omitted so -fsycl propagates to every TU and to consumers.
    add_sycl_to_target(TARGET occl_gtest_device)

    # occl_on_device - the correctness baseline. test_main.cpp carries
    # MPI_Init_thread before InitGoogleTest, so the binary links gtest, not
    # gtest_main.
    add_executable(occl_on_device
        suites/on_device/allreduce_test.cpp
        suites/on_device/collective_support_consistency_test.cpp
        suites/on_device/abort_listener_test.cpp
        suites/on_device/unanimous_agreement_test.cpp
        framework/test_main.cpp
    )
    target_link_libraries(occl_on_device PRIVATE occl_gtest_device gtest)
    target_compile_options(occl_on_device PRIVATE "${COVERAGE_CXX_FLAGS}" "${COMMON_CXX_FLAGS}")
    target_link_libraries(occl_on_device PRIVATE "${COVERAGE_CXX_FLAGS}")
    add_sycl_to_target(TARGET occl_on_device)

    # The entries pin the legacy plugin, so they exist only where it is built.
    # Every entry is self-contained: the plugin, its directory and the kernel
    # sources travel with it, so a ctest run needs nothing sourced first. The
    # BMG driver keys are not exported: they cost enough device memory to break
    # large runs, and only a run that reports nonzero #wrong needs them.
    if(NOT TARGET ccl_legacy)
        message(VERBOSE "occl_on_device: ccl_legacy is not built, so no ctest entries are registered")
    else()
        set(ON_DEVICE_ENV
            "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>;")
        string(APPEND ON_DEVICE_ENV
            "CCL_KERNEL_PATH=set:${PROJECT_SOURCE_DIR}/deps/libccl/src/kernels/;")
        string(APPEND ON_DEVICE_ENV "CCL_PLUGIN=set:ONECCL_LEGACY;")

        # A filter that matches nothing, or rows all skipped, still exits 0.
        # gtest then prints "[  PASSED  ] 0 tests", so that line fails the
        # entry.
        set(ON_DEVICE_EMPTY_RUN "\\[  PASSED  \\] 0 tests")

        # One entry per row set — two count lists and the in-place rows — so
        # "the remainder rows went red and the aligned rows did not" is
        # readable from the ctest report.
        foreach(row_set AlignedCounts RemainderCounts InPlace)
            set(entry "occl_on_device_${row_set}_AllReduce")
            add_test(NAME ${entry}
                COMMAND mpiexec -n 2 ${ONECCL_TEST_LAUNCHER}
                        $<TARGET_FILE:occl_on_device>
                        --gtest_filter=${row_set}_AllReduce/*
                CONFIGURATIONS unit_tests
            )
            set_property(TEST ${entry} PROPERTY ENVIRONMENT_MODIFICATION "${ON_DEVICE_ENV}")
            set_tests_properties(${entry} PROPERTIES LABELS "unit_tests" TIMEOUT 600
                FAIL_REGULAR_EXPRESSION "${ON_DEVICE_EMPTY_RUN}")
        endforeach()

        # Pins that ExecutorFactory's and ValidityTable's independent "supported
        # collectives" lists agree. Read-only (never calls setup()/execute()), so
        # it runs safely alongside the other -n 2 entries.
        add_test(NAME occl_on_device_CollectiveSupportConsistency
            COMMAND mpiexec -n 2 ${ONECCL_TEST_LAUNCHER}
                    $<TARGET_FILE:occl_on_device>
                    --gtest_filter=CollectiveSupportConsistencyTest.*
            CONFIGURATIONS unit_tests
        )
        set_property(TEST occl_on_device_CollectiveSupportConsistency PROPERTY
            ENVIRONMENT_MODIFICATION "${ON_DEVICE_ENV}")
        set_tests_properties(occl_on_device_CollectiveSupportConsistency PROPERTIES
            LABELS "unit_tests" TIMEOUT 600
            FAIL_REGULAR_EXPRESSION "${ON_DEVICE_EMPTY_RUN}")

        # Regression net for AbortListener's _Exit path. The death test forks;
        # its own -n 1 entry keeps the fork out of every other entry's process.
        add_test(NAME occl_on_device_AbortListenerDeathTest
            COMMAND mpiexec -n 1 ${ONECCL_TEST_LAUNCHER}
                    $<TARGET_FILE:occl_on_device>
                    --gtest_filter=AbortListenerDeathTest.*
            CONFIGURATIONS unit_tests
        )
        set_property(TEST occl_on_device_AbortListenerDeathTest PROPERTY
            ENVIRONMENT_MODIFICATION "${ON_DEVICE_ENV}")
        set_tests_properties(occl_on_device_AbortListenerDeathTest PROPERTIES
            LABELS "unit_tests" TIMEOUT 60
            FAIL_REGULAR_EXPRESSION "${ON_DEVICE_EMPTY_RUN}")

        add_test(NAME occl_on_device_UnanimousAgreement
            COMMAND mpiexec -n 2 ${ONECCL_TEST_LAUNCHER}
                    $<TARGET_FILE:occl_on_device>
                    --gtest_filter=UnanimousAgreementTest.*
            CONFIGURATIONS unit_tests
        )
        set_property(TEST occl_on_device_UnanimousAgreement PROPERTY
            ENVIRONMENT_MODIFICATION "${ON_DEVICE_ENV}")
        set_tests_properties(occl_on_device_UnanimousAgreement PROPERTIES
            LABELS "unit_tests" TIMEOUT 60
            FAIL_REGULAR_EXPRESSION "${ON_DEVICE_EMPTY_RUN}")
    endif()
endif()
