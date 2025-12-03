# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Version information that go into the generated config header
#------------------------------------------------------------------------------
set(GRETL_VERSION_MAJOR 0)
set(GRETL_VERSION_MINOR 0)
set(GRETL_VERSION_PATCH 1)
string(CONCAT GRETL_VERSION_FULL
    "v${GRETL_VERSION_MAJOR}"
    ".${GRETL_VERSION_MINOR}"
    ".${GRETL_VERSION_PATCH}" )

if (Git_FOUND)
  ## check to see if we are building from a Git repo or an exported tarball
  blt_is_git_repo( OUTPUT_STATE is_git_repo )

  if(${is_git_repo})
    blt_git_hashcode(HASHCODE sha1 RETURN_CODE rc)
    if(NOT ${rc} EQUAL 0)
      message(FATAL_ERROR "blt_git_hashcode failed!")
    endif()

    set(GRETL_GIT_SHA ${sha1})
  endif()

endif()

message(STATUS "Configuring Gretl version ${GRETL_VERSION_FULL}")


#--------------------------------------------------------------------------
# Add define we can use when debug builds are enabled
#--------------------------------------------------------------------------
set(GRETL_DEBUG FALSE)
if(CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)")
    set(GRETL_DEBUG TRUE)

    # Controls various behaviors in Axom, like turning off/on SLIC debug and assert macros
    set(AXOM_DEBUG TRUE)
endif()


#------------------------------------------------------------------------------
# General Build Info
#------------------------------------------------------------------------------
gretl_convert_to_native_escaped_file_path(${PROJECT_SOURCE_DIR} GRETL_REPO_DIR)
gretl_convert_to_native_escaped_file_path(${CMAKE_BINARY_DIR}   GRETL_BINARY_DIR)

#------------------------------------------------------------------------------
# Create Config Header
#------------------------------------------------------------------------------
gretl_configure_file(
    ${PROJECT_SOURCE_DIR}/src/gretl/config.hpp.in
    ${PROJECT_BINARY_DIR}/include/gretl/config.hpp
)

install(FILES ${PROJECT_BINARY_DIR}/include/gretl/config.hpp DESTINATION include/gretl)

#------------------------------------------------------------------------------
# Generate gretl-config.cmake for importing Gretl into other CMake packages
#------------------------------------------------------------------------------

# Set up some paths, preserve existing cache values (if present)
set(GRETL_INSTALL_INCLUDE_DIR "include" CACHE STRING "")
set(GRETL_INSTALL_CONFIG_DIR "lib" CACHE STRING "")
set(GRETL_INSTALL_LIB_DIR "lib" CACHE STRING "")
set(GRETL_INSTALL_BIN_DIR "bin" CACHE STRING "")
set(GRETL_INSTALL_CMAKE_MODULE_DIR "${GRETL_INSTALL_CONFIG_DIR}/cmake" CACHE STRING "")

set(GRETL_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)


include(CMakePackageConfigHelpers)

# Add version helper
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gretl-config-version.cmake
    VERSION ${GRETL_VERSION_FULL}
    COMPATIBILITY AnyNewerVersion
)

# Set up cmake package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gretl-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gretl-config.cmake
  INSTALL_DESTINATION
    ${GRETL_INSTALL_CONFIG_DIR}
  PATH_VARS
    GRETL_INSTALL_INCLUDE_DIR
    GRETL_INSTALL_LIB_DIR
    GRETL_INSTALL_BIN_DIR
    GRETL_INSTALL_CMAKE_MODULE_DIR
  )

# Install config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gretl-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gretl-config-version.cmake
  DESTINATION
    ${GRETL_INSTALL_CMAKE_MODULE_DIR}
)

# Install BLT files that recreate BLT targets in downstream projects
blt_install_tpl_setups(DESTINATION ${GRETL_INSTALL_CMAKE_MODULE_DIR})
