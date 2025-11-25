# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option(ENABLE_ASAN "Enable AddressSanitizer for memory checking (Clang or GCC only)" OFF)
if(ENABLE_ASAN)
    if(NOT (C_COMPILER_FAMILY_IS_CLANG OR C_COMPILER_FAMILY_IS_GNU))
        message(FATAL_ERROR "ENABLE_ASAN only supports Clang and GCC")
    endif()
endif()

# Only enable Gretl's code checks by default if it is the top-level project
# or a user overrides it
if("${CMAKE_PROJECT_NAME}" STREQUAL "gretl")
    set(_enable_gretl_code_checks ON)
else()
    set(_enable_gretl_code_checks OFF)
endif()
option(GRETL_ENABLE_CODE_CHECKS "Enable Gretl's code checks" ${_enable_gretl_code_checks})

cmake_dependent_option(GRETL_ENABLE_TESTS "Enables Gretl Tests" ON "ENABLE_TESTS" OFF)

