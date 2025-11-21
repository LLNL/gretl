# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

cmake_minimum_required(VERSION 3.16)
project(gretl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

file(GLOB_RECURSE cpp_sources 
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/*.hpp)
  
add_library(gretl ${cpp_sources})

target_include_directories(gretl PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

file(GLOB_RECURSE gretl_headers 
  ${PROJECT_SOURCE_DIR}/src/*.hpp)

# Specify public headers for the library
set_target_properties(gretl PROPERTIES PUBLIC_HEADER gretl_headers)

# Install the library and its public headers
install(TARGETS gretl
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gretl)

if (GRETL_ENABLE_ASAN)
  add_link_options(-fsanitize=address)
  add_compile_options(-fsanitize=address)
endif()

if (GRETL_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

