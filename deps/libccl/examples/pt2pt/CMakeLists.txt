file(GLOB sources "./src/*.c" "./src/*.cpp")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include_directories(include)
include_directories(src)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})

    target_include_directories(${executable} PRIVATE ${EXAMPLES_INC_DIRS})
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC pthread)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PUBLIC MPI::MPI_CXX)
    target_link_libraries(${executable} PUBLIC ${COMPUTE_BACKEND_TARGET_NAME})
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_EXAMPLES}/pt2pt OPTIONAL)
endforeach()
