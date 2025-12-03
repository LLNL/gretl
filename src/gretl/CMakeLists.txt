# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# This file contains metadata that often changes (ie. git sha).
# Hide these changes in the source file so we do not force
# a full rebuild.
gretl_configure_file(
  git_sha.hpp.in
  ${PROJECT_BINARY_DIR}/include/gretl/git_sha.hpp
 )
 
set(gretl_sources
    about.cpp
    data_store.cpp
    state_base.cpp
    vector_state.cpp)

set(gretl_headers
    about.hpp
    checkpoint.hpp
    create_state.hpp
    data_store.hpp
    double_state.hpp
    ${PROJECT_BINARY_DIR}/include/gretl/git_sha.hpp
    print_utils.hpp
    state_base.hpp
    state.hpp
    test_utils.hpp
    upstream_state.hpp
    vector_state.hpp)
  
blt_add_library(NAME    gretl
                SOURCES ${gretl_sources}
                HEADERS ${gretl_headers})

gretl_write_unified_header(
    NAME    gretl
    HEADERS ${gretl_headers}
    )

target_include_directories(gretl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${gretl_headers} ${PROJECT_BINARY_DIR}/include/gretl/gretl.hpp 
        DESTINATION include/gretl )

install(TARGETS              gretl
        EXPORT               gretl-targets
        DESTINATION          lib
        )

