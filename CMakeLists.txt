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

target_include_directories(gretl PUBLIC ${PROJECT_SOURCE_DIR}/src)


if (gretl_ENABLE_ASAN)
  add_link_options(-fsanitize=address)
  add_compile_options(-fsanitize=address)
endif()

if (gretl_ENABLE_TESTING)
  add_subdirectory(tests)
endif()


