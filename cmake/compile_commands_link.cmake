#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

# Points <repo>/compile_commands.json at this build's database so git clang-check
# and IDE tooling (clangd) find it, including out-of-tree builds. Never replaces a
# real file someone put there; the most recently configured build wins. The repo
# root comes from this file's location, so it also holds under add_subdirectory().
function(oneccl_link_compile_commands)
    get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    set(_link "${_root}/compile_commands.json")
    if(CMAKE_BINARY_DIR STREQUAL _root)
        return()
    endif()
    if(EXISTS "${_link}" AND NOT IS_SYMLINK "${_link}")
        message(STATUS "oneCCL: keeping existing ${_link} (not a symlink)")
        return()
    endif()
    file(REMOVE "${_link}")
    file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${_link}" SYMBOLIC RESULT _result)
    if(_result EQUAL 0)
        message(STATUS "oneCCL: compile_commands.json -> ${CMAKE_BINARY_DIR}")
    else()
        message(WARNING "oneCCL: could not link compile_commands.json: ${_result}")
    endif()
endfunction()
