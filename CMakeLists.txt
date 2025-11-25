# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

cmake_minimum_required(VERSION 3.16)
project(gretl LANGUAGES C CXX)

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------

if(DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Tune BLT to our needs
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
endif()

# These BLT tools are not used in Gretl, turn them off
set(_unused_blt_tools
    CLANGQUERY
    CLANGTIDY
    DOXYGEN
    SPHINX
    VALGRIND
    ASTYLE
    CMAKEFORMAT
    UNCRUSTIFY
    YAPF)
foreach(_tool ${_unused_blt_tools})
    set(ENABLE_${_tool} OFF CACHE BOOL "")
endforeach()

# These BLT tools are only used by Gretl developers, so turn them off
# unless an explicit executable path is given
set(_used_blt_tools
    CLANGFORMAT
    CPPCHECK)
foreach(_tool ${_used_blt_tools})
    if(NOT ${_tool}_EXECUTABLE)
        set(ENABLE_${_tool} OFF CACHE BOOL "")
    else()
        set(ENABLE_${_tool} ON CACHE BOOL "")
    endif()
endforeach()

set(BLT_REQUIRED_CLANGFORMAT_VERSION  "19" CACHE STRING "")

set(ENABLE_ALL_WARNINGS ON CACHE BOOL "")
set(ENABLE_WARNINGS_AS_ERRORS ON CACHE BOOL "")

set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")
if (BLT_EXPORT_THIRDPARTY)
    message(FATAL_ERROR "Gretl does not use BLT's exporting of TPLs, it conflicts with the new blt_install_tpl_setups macro.")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------
# Setup Macros/Basics
#------------------------------------------------------------------------------

include(${PROJECT_SOURCE_DIR}/cmake/GretlMacros.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/GretlBasics.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/GretlCompilerFlags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/GretlConfigHeader.cmake)

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

message(STATUS "Gretl Code Checks: ${GRETL_ENABLE_CODE_CHECKS}")

if (GRETL_ENABLE_CODE_CHECKS)
    gretl_add_code_checks(PREFIX gretl)
endif()


#------------------------------------------------------------------------------
# Add subdirectories
#------------------------------------------------------------------------------

add_subdirectory(src)


#------------------------------------------------------------------------------
# Export Targets
#------------------------------------------------------------------------------

install(EXPORT gretl-targets 
        NAMESPACE gretl::
        DESTINATION lib/cmake
        )
