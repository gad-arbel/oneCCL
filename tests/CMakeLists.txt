include(CMakeParseArguments)

function(add_test_internal)
    cmake_parse_arguments(
        ARGS # Prefix for parsed arguments
        "" # Boolean arguments (none in this case)
        "TARGET;EXEC_PREFIX;TIMEOUT" # Mono-valued arguments
        "SRCS;PLUGIN_TARGETS;VALUATIONS" # Multi-valued arguments
        ${ARGN}
    )

    # Ensure that TARGET and SRCS are provided as they are essential
    if(NOT ARGS_TARGET)
        message(FATAL_ERROR "A TARGET must be specified.")
    endif()
    if(NOT ARGS_SRCS)
        message(FATAL_ERROR "Source files must be specified in SRCS.")
    endif()

    # Check PLUGIN_TARGETS availability
    foreach(plugin_target ${ARGS_PLUGIN_TARGETS})
        if(NOT TARGET ${plugin_target})
            message(WARNING "Skipping test for ${ARGS_TARGET}: required plugin target '${plugin_target}' is not available.")
            return()
        endif()
    endforeach()

    # Add an executable for the test
    add_executable(${ARGS_TARGET} ${ARGS_SRCS})

    # Find threads
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    # Find the MPI package
    if(ONECCL_TEST_USE_MPI)
        find_package(MPI REQUIRED)
    endif()

    # Link necessary libraries
    set(LIBRARIES_TO_LINK
        oneCCL2
        ccl_null
        gtest
        gtest_main
        Threads::Threads
    )

    if(ONECCL_TEST_USE_MPI)
        list(APPEND LIBRARIES_TO_LINK MPI::MPI_CXX)
    endif()

    if(ONECCL_ENABLE_ITT)
        list(APPEND LIBRARIES_TO_LINK ittnotify)
    endif()

    target_link_libraries(${ARGS_TARGET} PRIVATE ${LIBRARIES_TO_LINK})

    target_include_directories(${ARGS_TARGET} PRIVATE "${PROJECT_SOURCE_DIR}/src")

    # Add coverage
    target_compile_options(${ARGS_TARGET} PRIVATE "${COVERAGE_CXX_FLAGS}")
    target_link_libraries(${ARGS_TARGET} PRIVATE "${COVERAGE_CXX_FLAGS}")

    # Add common compatiblity flags
    target_compile_options(${ARGS_TARGET} PRIVATE "${COMMON_CXX_FLAGS}")

    # Split EXEC_PREFIX into individual arguments, like "mpiexec -n 2" into "mpiexec" "-n" "2"
    if(ARGS_EXEC_PREFIX)
        separate_arguments(ARGS_EXEC_PREFIX)
    endif()

    # Prepare the command with execution prefix if specified
    set(test_command ${ARGS_EXEC_PREFIX} ${ONECCL_TEST_LAUNCHER} $<TARGET_FILE:${ARGS_TARGET}>)
    add_test(NAME ${ARGS_TARGET} COMMAND ${test_command} CONFIGURATIONS unit_tests)

    # Handle Plugin Targets
    foreach(plugin_target ${ARGS_PLUGIN_TARGETS})
        set(ENV_MODIFICATION "${ENV_MODIFICATION}LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:${plugin_target}>;")
    endforeach()

    set_property(TEST ${ARGS_TARGET} PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    set_tests_properties(${ARGS_TARGET} PROPERTIES TIMEOUT ${ARGS_TIMEOUT})

    if(${ONECCL_VALGRIND_TESTING} AND ${ARGS_VALUATIONS})
        add_test(NAME ${ARGS_TARGET}_valgrind COMMAND valgrind --tool=memcheck --leak-check=full --show-reachable=yes --error-exitcode=1 ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${ARGS_TARGET}_valgrind PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()

    if(${ONECCL_HELGRIND_TESTING} AND ${ARGS_VALUATIONS})
        add_test(NAME ${ARGS_TARGET}_helgrind COMMAND valgrind --tool=helgrind --error-exitcode=1 ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${ARGS_TARGET}_helgrind PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()
endfunction()

add_test_internal(
    TARGET version_api
    SRCS version_api.cpp
    PLUGIN_TARGETS "ccl_null"
    VALUATIONS ON
    EXEC_PREFIX ""
    TIMEOUT 10
)

add_test_internal(
    TARGET null_smoke
    SRCS null_api.cpp
    PLUGIN_TARGETS "ccl_null"
    VALUATIONS ON
    EXEC_PREFIX ""
    TIMEOUT 30
)

if(NOT WIN32)
    # The GPU plugin is optional: builds without SYCL have only the CPU one
    set(plugin_selection_plugins "ccl_legacy_cpu;ccl_null")
    if(TARGET ccl_legacy)
        list(PREPEND plugin_selection_plugins ccl_legacy)
    endif()
    add_test_internal(
        TARGET plugin_selection
        SRCS plugin_selection.cpp
        PLUGIN_TARGETS "${plugin_selection_plugins}"
        VALUATIONS OFF
        EXEC_PREFIX ""
        TIMEOUT 60
    )
    if(TARGET plugin_selection)
        # Selected by its path, so not named libccl_*
        add_library(plugin_selection_fake SHARED plugin_selection_fake.cpp)
        target_include_directories(plugin_selection_fake PRIVATE "${PROJECT_SOURCE_DIR}/src")
        target_link_libraries(plugin_selection_fake PRIVATE oneCCL2)
        target_compile_options(plugin_selection_fake PRIVATE "${COMMON_CXX_FLAGS}")
        add_dependencies(plugin_selection plugin_selection_fake)

        target_link_libraries(plugin_selection PRIVATE ${CMAKE_DL_LIBS})
        target_compile_definitions(plugin_selection PRIVATE
            ONECCL_TEST_FAKE_PLUGIN_PATH="$<TARGET_FILE:plugin_selection_fake>")
        if(TARGET ccl_legacy)
            target_compile_definitions(plugin_selection PRIVATE
                ONECCL_TEST_GPU_PLUGIN_PATH="$<TARGET_FILE:ccl_legacy>")
        endif()
    endif()

    # Only the null plugin in the search path: no plugin is selected
    add_test_internal(
        TARGET plugin_selection_null_only
        SRCS plugin_selection_null_only.cpp
        PLUGIN_TARGETS "ccl_null"
        VALUATIONS OFF
        EXEC_PREFIX ""
        TIMEOUT 30
    )
endif()

# Temporarly disenable valgrind checks for the test
if(ONECCL_TEST_USE_MPI)
    add_test_internal(
        TARGET null_concurrent
        SRCS concurrent_execution.cpp
        PLUGIN_TARGETS "ccl_null"
        VALUATIONS OFF
        EXEC_PREFIX "mpiexec -n 2"
        TIMEOUT 30
    )
endif()

if(COMPILER_SUPPORTS_SYCL AND ONECCL_TEST_USE_MPI)
    # Build the acceptance_sycl binary once
    add_executable(acceptance_sycl acceptance_sycl.cpp)

    set(LIBRARIES_TO_LINK
        oneCCL2
        ccl_null
        gtest
        gtest_main
        MPI::MPI_CXX
        Threads::Threads
    )

    if(ONECCL_ENABLE_ITT)
        list(APPEND LIBRARIES_TO_LINK ittnotify)
    endif()

    target_link_libraries(acceptance_sycl PRIVATE ${LIBRARIES_TO_LINK})
    target_include_directories(acceptance_sycl PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_compile_options(acceptance_sycl PRIVATE "${COVERAGE_CXX_FLAGS}" "${COMMON_CXX_FLAGS}")
    target_link_libraries(acceptance_sycl PRIVATE "${COVERAGE_CXX_FLAGS}")
    add_sycl_to_target(TARGET acceptance_sycl SOURCES acceptance_sycl.cpp)

    # Prepare environment for all acceptance_sycl tests
    if(TARGET ccl_legacy)
        set(ACCEPTANCE_ENV "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_legacy>;")
        set(ACCEPTANCE_ENV "${ACCEPTANCE_ENV}CCL_KERNEL_PATH=set:${PROJECT_SOURCE_DIR}/deps/libccl/src/kernels/;")
        set(ACCEPTANCE_ENV "${ACCEPTANCE_ENV}CCL_PLUGIN=set:ONECCL_LEGACY;")
    endif()

    # Split into separate tests by collective operation
    set(COLLECTIVE_TESTS
        AllReduceOperation
        ReduceOperation
        BroadcastOperation
        AllGatherOperation
        AllToAllOperation
        AllToAllvOperation
        ReduceScatterOperation
        ScatterOperation
        SendRecvOperation
        SendRecvGroupOperation
    )

    set(SIMPLE_TESTS
        CustomRedopAllReduceOperation
        SplitOperation
        BlockingComm
        BufferRegistrationApi
        WindowRegistrationApi
        FinalizeAfterCollectives
        FinalizeWithGroupAPI
        FinalizeWithGroupSendRecv
        GraphWithManualSync
        FinalizeAfterGraphRecording
        FinalizeMultipleQueuesInOrder
        FinalizeDuringGroupThrows
    )

    # Add test for each collective operation
    foreach(test_name ${COLLECTIVE_TESTS})
        set(full_test_name "acceptance_sycl_${test_name}")
        set(test_command mpiexec -n 2 ${ONECCL_TEST_LAUNCHER} $<TARGET_FILE:acceptance_sycl> --gtest_filter=*CollectiveTest.${test_name}*)
        add_test(NAME ${full_test_name} COMMAND ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${full_test_name} PROPERTY ENVIRONMENT_MODIFICATION "${ACCEPTANCE_ENV}")
        set_tests_properties(${full_test_name} PROPERTIES TIMEOUT 600)
    endforeach()

    if(TARGET ccl_legacy)
        # For two ranks, 4 MiB is above the default small-message threshold but
        # below AUTO's 16 MiB ring boundary. AUTO must therefore keep the large
        # dispatcher and select IPC instead of redirecting the call to RT256.
        add_test(
            NAME broadcast_auto_fallback_to_ipc
            COMMAND
                bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_and_expect_output.sh
                "broadcast path: ipc"
                mpiexec -n 2
                ${ONECCL_TEST_LAUNCHER}
                $<TARGET_FILE:acceptance_sycl>
                --gtest_filter=BroadcastAutoFallbackTest.FourMiBInt32
            CONFIGURATIONS unit_tests
        )
        set(broadcast_auto_fallback_env "${ACCEPTANCE_ENV}")
        string(APPEND broadcast_auto_fallback_env "CCL_LOG_LEVEL=set:debug;")
        string(APPEND broadcast_auto_fallback_env "CCL_SYCL_FORCE_PCIE=set:1;")
        string(APPEND broadcast_auto_fallback_env "CCL_SYCL_BROADCAST_TMP_BUF=set:0;")
        string(APPEND broadcast_auto_fallback_env "CCL_SYCL_BROADCAST_SMALL_THRESHOLD=set:524288;")
        string(APPEND broadcast_auto_fallback_env "CCL_SYCL_BROADCAST_RING=set:1;")
        string(APPEND broadcast_auto_fallback_env "CCL_SYCL_BROADCAST_RING_AUTO=set:1;")
        set_property(
            TEST broadcast_auto_fallback_to_ipc
            PROPERTY ENVIRONMENT_MODIFICATION "${broadcast_auto_fallback_env}"
        )
        set_tests_properties(
            broadcast_auto_fallback_to_ipc
            PROPERTIES
                PROCESSORS 2
                TIMEOUT 600
        )
    endif()

    # ChunkingTest under CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE and the ring segment variables. libccl
    # reads them once per process, so each setting is its own test. EXPECTED is text the output must
    # contain, or "" for none; ENV is a list of ENVIRONMENT_MODIFICATION entries ("\;" inside a value).
    # In the uncommon suite, so only the nightly and test:uncommon run them, not every PR.
    # The optional fifth argument is the number of ranks (default 2).
    function(add_chunking_test name filter expected env)
        set(ranks 2)
        if(ARGC GREATER 4)
            set(ranks ${ARGV4})
        endif()
        set(cmd mpiexec -n ${ranks} ${ONECCL_TEST_LAUNCHER}
                $<TARGET_FILE:acceptance_sycl> --gtest_filter=${filter})
        if(NOT expected STREQUAL "")
            set(cmd bash ${CMAKE_CURRENT_SOURCE_DIR}/run_and_expect_output.sh "${expected}" ${cmd})
        endif()
        add_test(NAME ${name} COMMAND ${cmd} CONFIGURATIONS uncommon)
        set_property(TEST ${name} PROPERTY ENVIRONMENT_MODIFICATION "${ACCEPTANCE_ENV}${env}")
        set_tests_properties(${name} PROPERTIES PROCESSORS ${ranks} TIMEOUT 900)
    endfunction()

    set(chunk_vars ALLREDUCE REDUCE_SCATTER ALLGATHERV ALLTOALL)
    set(segment_vars ALLREDUCE REDUCE_SCATTER ALLGATHERV)
    set(unaligned_env "")
    set(ranges_env "")
    set(ze_ce_env "CCL_ENABLE_ZE_CE=set:1;")
    foreach(coll ${chunk_vars})
        # odd, so not a multiple of any element size
        string(APPEND unaligned_env "CCL_SYCL_${coll}_SIMPLE_CHUNK_SIZE=set:1000001;")
        string(APPEND ranges_env "CCL_SYCL_${coll}_SIMPLE_CHUNK_SIZE=set:0:0-4194304\;262144:4194305-max;")
        # more than 500 chunks per call for every tested type and count
        string(APPEND ze_ce_env "CCL_SYCL_${coll}_SIMPLE_CHUNK_SIZE=set:8192;")
    endforeach()
    foreach(coll ${segment_vars})
        # not a multiple of the 16-byte vector
        string(APPEND unaligned_env "CCL_SYCL_${coll}_SIMPLE_RING_SEGMENT_SIZE=set:1000004;")
    endforeach()

    add_chunking_test(acceptance_sycl_chunking_default "*ChunkingTest.*" "" "")
    add_chunking_test(acceptance_sycl_chunking_unaligned "*ChunkingTest.*" "" "${unaligned_env}")
    add_chunking_test(acceptance_sycl_chunking_ranges "*ChunkingTest.*" "" "${ranges_env}")
    add_chunking_test(acceptance_sycl_chunking_ze_ce "*ChunkingTest.*" "" "${ze_ce_env}")

    # CE/EU pipeline: each test expects the pipeline log line of its collective. The smallest
    # chunk makes the ChunkingTest sizes take it. 2 ranks: BMG CI; 4 ranks: 8-GPU node (tests.yaml).
    set(ce_eu_env "CCL_SYCL_COPY_ENGINE=set:1;CCL_SYCL_CE_EU_PIPELINE=set:1;CCL_LOG_LEVEL=set:info;")
    set(ce_eu_chunk_env "${ce_eu_env}CCL_SYCL_CE_EU_PIPELINE_CHUNK_SIZE=set:65536;")
    foreach(ranks 2 4)
        if(ranks EQUAL 2)
            set(ce_eu_suffix "")
            set(ce_eu_label bmg_ci)
        else()
            set(ce_eu_suffix "_${ranks}r")
            set(ce_eu_label bmg_8gpu_ci)
        endif()
        foreach(coll allreduce reduce_scatter)
            if(coll STREQUAL "allreduce")
                set(ce_eu_gtest AllReduce)
            else()
                set(ce_eu_gtest ReduceScatter)
            endif()
            set(ce_eu_expected "collective=${coll} ranks=${ranks} selected_path=ce_eu_pipeline")
            set(ce_eu_tests acceptance_sycl_ce_eu_pipeline${ce_eu_suffix}_${coll})
            add_chunking_test(acceptance_sycl_ce_eu_pipeline${ce_eu_suffix}_${coll}
                              "*ChunkingTest.${ce_eu_gtest}*" "${ce_eu_expected}"
                              "${ce_eu_chunk_env}" ${ranks})
            # 2 ranks only: 4-rank bf16 sums miss the test tolerance, also without the pipeline
            if(ranks EQUAL 2)
                add_chunking_test(acceptance_sycl_ce_eu_pipeline_collectives_${coll}
                                  "*CollectiveTest.${ce_eu_gtest}Operation*" "${ce_eu_expected}"
                                  "${ce_eu_env}" ${ranks})
                list(APPEND ce_eu_tests acceptance_sycl_ce_eu_pipeline_collectives_${coll})
            endif()
            set_tests_properties(${ce_eu_tests} PROPERTIES LABELS ${ce_eu_label} TIMEOUT 300)
        endforeach()
    endforeach()
    set(invalid_var CCL_SYCL_ALLREDUCE_SIMPLE_CHUNK_SIZE)
    # marks the runs whose collective must be rejected; the test skips elsewhere
    set(expect_reject "ONECCL_TEST_EXPECT_CHUNK_REJECT=set:1;")
    foreach(bytes 1 2 3)
        add_chunking_test(acceptance_sycl_chunking_invalid_small_${bytes} "*ChunkingInvalidTest.*"
                          "${invalid_var} chunk size must be 0 or at least 8 bytes"
                          "${invalid_var}=set:${bytes};${expect_reject}")
    endforeach()
    add_chunking_test(acceptance_sycl_chunking_invalid_overflow "*ChunkingInvalidTest.*"
                      "${invalid_var} chunk size does not fit in 64 bits"
                      "${invalid_var}=set:99999999999999999999;${expect_reject}")
    add_chunking_test(acceptance_sycl_chunking_invalid_gap "*ChunkingInvalidTest.*"
                      "${invalid_var} has no chunk size for sizes 1048577-16777215"
                      "${invalid_var}=set:0:0-1048576\;4096:16777216-max;${expect_reject}")

    # self_copy_threshold, copy_path, simple_ring_segment_size and simple_chunk_size set a different
    # variable for some collectives. The policy gives each collective its own values; each knob line
    # must name that collective's variables and values. allreduce's 32 MiB call must take the
    # msg_size<=33554432 band, not the < band before it or the band after it. The load must warn about
    # the misspelled knob and the invalid copy_path, and reduce_scatter's simple_chunk_size must beat
    # the variable set below.
    # Instance 92 is float32, sum, 8M elements, large enough to select the large algorithm.
    set(policy_filter "*CollectiveTest.AllReduceOperation/92")
    string(APPEND policy_filter ":*CollectiveTest.ReduceScatterOperation/92")
    string(APPEND policy_filter ":*CollectiveTest.AllGatherOperation/92")
    string(APPEND policy_filter ":*CollectiveTest.AllToAllOperation/92")
    string(APPEND policy_filter ":*CollectiveTest.AllToAllvOperation/92")
    set(policy_cmd mpiexec -n 2 ${ONECCL_TEST_LAUNCHER}
            $<TARGET_FILE:acceptance_sycl> --gtest_filter=${policy_filter})
    foreach(expected
            "allreduce knobs: copy_path=kernel (CCL_SYCL_COPY_ENGINE) simple_ring_segment_size=5242880 (CCL_SYCL_ALLREDUCE_SIMPLE_RING_SEGMENT_SIZE) simple_chunk_size=1048576 (CCL_SYCL_ALLREDUCE_SIMPLE_CHUNK_SIZE)"
            "reduce_scatter knobs: copy_path=kernel (CCL_SYCL_COPY_ENGINE) simple_ring_segment_size=6291456 (CCL_SYCL_REDUCE_SCATTER_SIMPLE_RING_SEGMENT_SIZE) simple_chunk_size=2097152 (CCL_SYCL_REDUCE_SCATTER_SIMPLE_CHUNK_SIZE)"
            "allgather knobs: self_copy_threshold=1000001 (CCL_SYCL_ALLGATHERV_SELF_COPY_THRESHOLD) copy_path=kernel (CCL_SYCL_COPY_ENGINE) simple_ring_segment_size=7340032 (CCL_SYCL_ALLGATHERV_SIMPLE_RING_SEGMENT_SIZE) simple_chunk_size=3145728 (CCL_SYCL_ALLGATHERV_SIMPLE_CHUNK_SIZE)"
            "alltoall knobs: self_copy_threshold=2000002 (CCL_SYCL_ALLTOALL_SELF_COPY_THRESHOLD) copy_path=copy_engine (CCL_SYCL_ALLTOALL_COPY_ENGINE) simple_chunk_size=4194304 (CCL_SYCL_ALLTOALL_SIMPLE_CHUNK_SIZE)"
            "alltoallv knobs: self_copy_threshold=3000003 (CCL_SYCL_ALLTOALLV_SELF_COPY_THRESHOLD) copy_path=default (CCL_SYCL_ALLTOALL_COPY_ENGINE)"
            "unknown knob 'simple_chunk_szie' under collective=allreduce"
            "invalid value 'bogus' for knob 'copy_path' under collective=allreduce")
        set(policy_cmd bash ${CMAKE_CURRENT_SOURCE_DIR}/run_and_expect_output.sh "${expected}" ${policy_cmd})
    endforeach()
    add_test(NAME acceptance_sycl_policy_knobs COMMAND ${policy_cmd} CONFIGURATIONS unit_tests)
    set(policy_env "${ACCEPTANCE_ENV}")
    string(APPEND policy_env "CCL_SELECTION_POLICY_PATH=set:${CMAKE_CURRENT_SOURCE_DIR}/policy_test.json;")
    string(APPEND policy_env "CCL_LOG_LEVEL=set:info;")
    string(APPEND policy_env "CCL_SYCL_REDUCE_SCATTER_SIMPLE_CHUNK_SIZE=set:0;")
    set_property(TEST acceptance_sycl_policy_knobs PROPERTY ENVIRONMENT_MODIFICATION "${policy_env}")
    set_tests_properties(acceptance_sycl_policy_knobs PROPERTIES PROCESSORS 2 TIMEOUT 900)

    # Add test for each simple collective operation
    foreach(test_name ${SIMPLE_TESTS})
        set(full_test_name "acceptance_sycl_${test_name}")
        set(test_command mpiexec -n 2 ${ONECCL_TEST_LAUNCHER} $<TARGET_FILE:acceptance_sycl> --gtest_filter=*SimpleCollectiveTest.${test_name}*)
        add_test(NAME ${full_test_name} COMMAND ${test_command} CONFIGURATIONS unit_tests)
        set_property(TEST ${full_test_name} PROPERTY ENVIRONMENT_MODIFICATION "${ACCEPTANCE_ENV}")
        set_tests_properties(${full_test_name} PROPERTIES TIMEOUT 300)
    endforeach()

    # Scale-up coverage for the OCCP bootstrap (CCL_KVS_MODE=occp), which is
    # opt-in and therefore not touched by any of the tests above. One collective
    # is enough: the bootstrap channel is chosen once per communicator, so the
    # collective that runs on top of it only has to prove the communicator came
    # up and works. The multi-node half of OCCP is covered by the scaleout
    # examples leg and by sycl/scaleout/kvs_init_occp_scaleout.sh.
    set(full_test_name "acceptance_sycl_occp_AllReduceOperation")
    set(test_command mpiexec -n 2 ${ONECCL_TEST_LAUNCHER} $<TARGET_FILE:acceptance_sycl> --gtest_filter=*CollectiveTest.AllReduceOperation*)
    add_test(NAME ${full_test_name} COMMAND ${test_command} CONFIGURATIONS unit_tests)
    set_property(TEST ${full_test_name} PROPERTY ENVIRONMENT_MODIFICATION
                 "${ACCEPTANCE_ENV}CCL_KVS_MODE=set:occp;")
    set_tests_properties(${full_test_name} PROPERTIES TIMEOUT 600)

    add_test_internal(
        TARGET acceptance_thread_sycl
        SRCS acceptance_thread_sycl.cpp
        PLUGIN_TARGETS "ccl_legacy"
        VALUATIONS ON
        EXEC_PREFIX "mpiexec -n 1"
        TIMEOUT 600
    )

    if(TARGET acceptance_thread_sycl)
        add_sycl_to_target(TARGET acceptance_thread_sycl SOURCES acceptance_thread_sycl.cpp)

        get_property(ENV_MODIFICATION TEST acceptance_thread_sycl PROPERTY ENVIRONMENT_MODIFICATION)
        set(ENV_MODIFICATION "${ENV_MODIFICATION}CCL_KERNEL_PATH=set:${PROJECT_SOURCE_DIR}/deps/libccl/src/kernels/;")
        set_property(TEST acceptance_thread_sycl PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
    endif()

    if(TARGET ccl_legacy)
        add_executable(broadcast_ring_sycl broadcast_ring_sycl.cpp)
        target_link_libraries(
            broadcast_ring_sycl
            PRIVATE
                oneCCL2
                ccl_null
                MPI::MPI_CXX
                Threads::Threads
        )
        target_compile_options(
            broadcast_ring_sycl
            PRIVATE
                "${COVERAGE_CXX_FLAGS}"
                "${COMMON_CXX_FLAGS}"
        )
        target_link_libraries(broadcast_ring_sycl PRIVATE "${COVERAGE_CXX_FLAGS}")
        add_sycl_to_target(TARGET broadcast_ring_sycl SOURCES broadcast_ring_sycl.cpp)

        function(
            add_bmg_broadcast_ring_test
            test_name
            scenario
            rank_count
            ci_label
            expected_output
        )
            set(test_env "${ACCEPTANCE_ENV}")
            foreach(env_modification IN LISTS ARGN)
                string(APPEND test_env "${env_modification};")
            endforeach()

            add_test(
                NAME ${test_name}
                COMMAND
                    bash
                    ${CMAKE_CURRENT_SOURCE_DIR}/run_and_expect_output.sh
                    ${expected_output}
                    mpiexec -n ${rank_count}
                    ${ONECCL_TEST_LAUNCHER}
                    $<TARGET_FILE:broadcast_ring_sycl>
                    ${scenario}
                CONFIGURATIONS unit_tests
            )
            set_property(TEST ${test_name} PROPERTY ENVIRONMENT_MODIFICATION "${test_env}")
            set_tests_properties(
                ${test_name}
                PROPERTIES
                    LABELS "${ci_label};broadcast_ring"
                    PROCESSORS ${rank_count}
                    TIMEOUT 600
            )
        endfunction()

        set(broadcast_ring_common_env
            "CCL_LOG_LEVEL=set:debug"
            "CCL_SYCL_BROADCAST_TMP_BUF=set:0"
            "CCL_SYCL_FORCE_PCIE=unset:"
        )

        add_bmg_broadcast_ring_test(
            broadcast_ring_auto_fallback
            auto_fallback
            2
            bmg_ci
            "broadcast path: ipc"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:1"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:1"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=unset:"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_auto
            auto_ring
            2
            bmg_ci
            "broadcast ring path: copy_engine"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:1"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:1"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=unset:"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_bmg_default
            bmg_default
            2
            bmg_ci
            "broadcast ring path: copy_engine"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=unset:"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:0"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:1"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=unset:"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_chunking
            chunking
            2
            bmg_ci
            "broadcast ring path: copy_engine, world=2, bytes=16777220, slice_bytes=4194304, slices=5"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:0"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:1"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=set:4194304"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_copy_kernel
            copy_kernel
            2
            bmg_ci
            "broadcast ring path: copy_kernel"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:0"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:0"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=set:0"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_relay_copy_engine
            relay_copy_engine
            8
            bmg_8gpu_ci
            "broadcast ring path: copy_engine, world=8"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:0"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:1"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=set:0"
        )
        add_bmg_broadcast_ring_test(
            broadcast_ring_relay_copy_kernel
            relay_copy_kernel
            8
            bmg_8gpu_ci
            "broadcast ring path: copy_kernel, world=8"
            ${broadcast_ring_common_env}
            "CCL_SYCL_BROADCAST_RING=set:1"
            "CCL_SYCL_BROADCAST_RING_AUTO=set:0"
            "CCL_SYCL_BROADCAST_RING_COPY_ENGINE=set:0"
            "CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD=set:0"
        )
    endif()
endif()

if(NOT ${ONECCL_USE_SYSTEM_LIBCCL})
    # This test is only enabled if we built libccl internally,
    # because it adds libccl.so.2 as dependency of libccl.so.1
    add_test(NAME both_apis COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/both_apis.sh ${CMAKE_CXX_COMPILER} CONFIGURATIONS unit_tests_host)

    # The null plugin is only used when selected explicitly
    set(ENV_MODIFICATION "${ENV_MODIFICATION}LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ccl_null>;CCL_PLUGIN=set:ONECCL_NULL;")
    set_property(TEST both_apis PROPERTY ENVIRONMENT_MODIFICATION "${ENV_MODIFICATION}")
endif()

if(COMPILER_SUPPORTS_SYCL AND ONECCL_TEST_USE_MPI)
    add_subdirectory(benchmark)
    add_subdirectory(rate_limiter)
endif()

# New gtest test infrastructure under correctness/ (guarded by ONECCL_BUILD_CORRECTNESS_TESTS)
if(ONECCL_BUILD_CORRECTNESS_TESTS)
    add_subdirectory(correctness/gtest)
endif()
