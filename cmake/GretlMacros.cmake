# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

##------------------------------------------------------------------------------
## Adds code checks for all cpp/hpp files recursively under the current directory
## that regex match INCLUDES and excludes any files that regex match EXCLUDES
## 
## This creates the following parent build targets:
##  check - Runs a non file changing style check and CppCheck
##  style - In-place code formatting
##
## Creates various child build targets that follow this pattern:
##  gretl_<check|style>
##  gretl_<cppcheck|clangformat>_<check|style>
##
## This also creates targets for running clang-tidy on the src/ and test/
## directories, with a more permissive set of checks for the tests,
## called gretl_guidelines_check and gretl_guidelines_check_tests, respectively
##------------------------------------------------------------------------------
macro(gretl_add_code_checks)

    set(options)
    set(singleValueArgs PREFIX)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create file globbing expressions that only include directories that contain source
    set(_base_dirs "tests" "src" "examples")
    # Note: any extensions added here should also be added to BLT's lists in CMakeLists.txt
    set(_ext_expressions "*.cpp" "*.hpp" "*.inl" "*.cuh" "*.cu" "*.cpp.in" "*.hpp.in")

    set(_glob_expressions)
    foreach(_exp ${_ext_expressions})
        foreach(_base_dir ${_base_dirs})
            list(APPEND _glob_expressions "${PROJECT_SOURCE_DIR}/${_base_dir}/${_exp}")
        endforeach()
    endforeach()

    # Glob for list of files to run code checks on
    set(_sources)
    file(GLOB_RECURSE _sources ${_glob_expressions})

    blt_add_code_checks(PREFIX          ${arg_PREFIX}
                        SOURCES         ${_sources}
                        CLANGFORMAT_CFG_FILE ${PROJECT_SOURCE_DIR}/.clang-format
                        CPPCHECK_FLAGS  --enable=all --inconclusive)


    set(_src_sources)
    file(GLOB_RECURSE _src_sources "src/*.cpp" "src/*.hpp" "src/*.inl")
    list(FILTER _src_sources EXCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*"
                              SRC_FILES         ${_src_sources})

    # Create list of recursive test directory glob expressions
    # NOTE: GLOB operator ** did not appear to be supported by cmake and did not recursively find test subdirectories
    # NOTE: Do not include all directories at root (for example: blt)

    file(GLOB_RECURSE _test_sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/tests/*.cpp")
    list(FILTER _test_sources INCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check_tests
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers"
                              SRC_FILES         ${_test_sources})
                                  
    if (ENABLE_COVERAGE)
        blt_add_code_coverage_target(NAME   ${arg_PREFIX}_coverage
                                     RUNNER ${CMAKE_MAKE_PROGRAM} test
                                     SOURCE_DIRECTORIES ${PROJECT_SOURCE_DIR}/src )
    endif()

endmacro(gretl_add_code_checks)


##------------------------------------------------------------------------------
## gretl_convert_to_native_escaped_file_path( path output )
##
## This macro converts a cmake path to a platform specific string literal
## usable in C++.  (For example, on windows C:/Path will be come C:\\Path)
##------------------------------------------------------------------------------
macro(gretl_convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro(gretl_convert_to_native_escaped_file_path)


##------------------------------------------------------------------------------
## gretl_remove_string_prefix
##
## This macro removes a string prefix from a given string.
##
## PREFIX - String prefix to be removed
## INPUT - String with possible prefix to be removed
## OUTPUT_VAR - Possibly altered output string
##
##------------------------------------------------------------------------------
macro(gretl_remove_string_prefix)
    set(options)
    set(singleValueArgs PREFIX INPUT OUTPUT_VAR)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_PREFIX)
         message(FATAL_ERROR "PREFIX is required")
    endif()

    if(NOT DEFINED arg_INPUT)
         message(FATAL_ERROR "INPUT is required")
    endif()

    if(NOT DEFINED arg_OUTPUT_VAR)
         message(FATAL_ERROR "OUTPUT_VAR is required")
    endif()

    string(LENGTH "${arg_PREFIX}" _prefix_len)
    string(SUBSTRING "${arg_INPUT}" 0 ${_prefix_len} _actual_prefix)

    if(_actual_prefix STREQUAL "${arg_PREFIX}")
        string(REPLACE "${arg_PREFIX}" "" _stripped_path "${arg_INPUT}")
        set(${arg_OUTPUT_VAR} "${_stripped_path}")
    else()
        set(${arg_OUTPUT_VAR} "${arg_INPUT}")
    endif()
endmacro()


##------------------------------------------------------------------------------
## gretl_configure_file
##
## This macro is a thin wrapper over the builtin configure_file command.
## It has the same arguments/options as configure_file but introduces an
## intermediate file that is only copied to the target file if the target differs
## from the intermediate.
##------------------------------------------------------------------------------
macro(gretl_configure_file _source _target)
    set(_tmp_target ${_target}.tmp)
    configure_file(${_source} ${_tmp_target} ${ARGN})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_target} ${_target})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_target})
endmacro(gretl_configure_file)


##------------------------------------------------------------------------------
## gretl_remove_string_prefix
##
## This macro fills OUTPUT_VAR with the subdirectory relative to 
## "<lower case project name>/src"
##
## OUTPUT_VAR - Possibly altered output string
##
##------------------------------------------------------------------------------
macro(gretl_get_src_subdirectory)

    set(options)
    set(singleValueArgs OUTPUT_VAR)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_OUTPUT_VAR)
         message(FATAL_ERROR "OUTPUT_VAR is required")
    endif()

    string(TOLOWER ${PROJECT_NAME} _project_name)

    # Convert both paths to absolute
    file(REAL_PATH "${PROJECT_SOURCE_DIR}/src/${_project_name}" abs_base_dir)
    file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" abs_full_path)

    # Check if full path starts with the base directory
    string(FIND "${abs_full_path}" "${abs_base_dir}" found_pos)

    if(NOT found_pos EQUAL 0)
        message(FATAL_ERROR "The full path '${abs_full_path}' does not start with the base directory '${abs_base_dir}'")
    endif()

    # Strip base directory from full path
    gretl_remove_string_prefix(PREFIX "${abs_base_dir}/"
                               INPUT "${abs_full_path}"
                               OUTPUT_VAR ${arg_OUTPUT_VAR})
endmacro(gretl_get_src_subdirectory)


##------------------------------------------------------------------------------
## gretl_write_unified_header
##
## This macro writes the unified header (<lowered PROJECT_NAME/<lowered NAME>.hpp)
## to the build directory for the given NAME with the given HEADERS included
## inside of it.
##
## NAME - The name of the unified header.
## HEADERS - Headers to be included in the header.
## NO_PATH_MODIFICATION - ON/OFF(default) Stops include path modification if on,
##    used for the project-level unified header
##
##------------------------------------------------------------------------------
# List to hold all unified headers to be later used to create a master unified header
set(_${PROJECT_NAME}_unified_headers "" CACHE STRING "" FORCE)
macro(gretl_write_unified_header)

    set(options)
    set(singleValueArgs NAME NO_PATH_MODIFICATION)
    set(multiValueArgs HEADERS EXCLUDE)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_NO_PATH_MODIFICATION)
        set(arg_NO_PATH_MODIFICATION OFF)
    endif()

    string(TOLOWER ${arg_NAME} _unified_header_name)
    string(TOLOWER ${PROJECT_NAME} _project_name)
    set(_header ${PROJECT_BINARY_DIR}/include/${_project_name}/${_unified_header_name}.hpp)
    set(_tmp_header ${_header}.tmp)
    set(_src_subdir)
    if(NOT arg_NO_PATH_MODIFICATION)
        gretl_get_src_subdirectory(OUTPUT_VAR _src_subdir)
    endif()

    file(WRITE ${_tmp_header} "\/\/ Copyright Lawrence Livermore National Security, LLC and
\/\/ other ${PROJECT_NAME} Project Developers. See the top-level LICENSE file for details.
\/\/
\/\/ SPDX-License-Identifier: (BSD-3-Clause)
\n
")

    file(APPEND ${_tmp_header} "#pragma once\n\n")

    file(APPEND ${_tmp_header} "#include \"${_project_name}\/${_project_name}_config.hpp\"\n\n")

    foreach(_file ${arg_HEADERS})
        if("${_file}" IN_LIST arg_EXCLUDE)
            continue()
        elseif("${_file}" MATCHES "\\.inl$")
            continue()
        elseif("${_file}" MATCHES "(\/detail\/)|(\/internal\/)")
            continue()
        elseif(arg_NO_PATH_MODIFICATION)
            file(APPEND ${_tmp_header} "#include \"${_file}\"\n")
        else()
            set(_headerPath)
            gretl_remove_string_prefix(PREFIX "${PROJECT_BINARY_DIR}\/"
                                       INPUT "${_file}"
                                       OUTPUT_VAR _headerPath)
            gretl_remove_string_prefix(PREFIX "include\/${_project_name}\/${arg_NAME}\/"
                                       INPUT "${_headerPath}"
                                       OUTPUT_VAR _headerPath)
            set(_headerPath "${_project_name}\/${_src_subdir}\/${_headerPath}")
            file(APPEND ${_tmp_header} "#include \"${_headerPath}\"\n")
        endif()
    endforeach()

    file(APPEND ${_tmp_header} "\n")

    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_header} ${_header})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_header})

    install(FILES       ${_header}
            DESTINATION include/${_project_name})

    # Add this component's unified header to the list to be added to the project specific unified header
    set(_component_header "${_project_name}/${_unified_header_name}.hpp")
    if("${_${PROJECT_NAME}_unified_headers}" STREQUAL "")
        set(_${PROJECT_NAME}_unified_headers "${_component_header}" CACHE STRING "" FORCE)
    else()
        set(_${PROJECT_NAME}_unified_headers "${_${PROJECT_NAME}_unified_headers};${_component_header}" CACHE STRING "" FORCE)
    endif()
endmacro(gretl_write_unified_header)
