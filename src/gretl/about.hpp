// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file about.hpp
 *
 * @brief This file contains the interface used for retrieving information
 * about how the driver is configured.
 */

#pragma once

#include <string>
#include <utility>

namespace gretl {

/**
 * @brief Returns a string about the configuration of Gretl
 *
 * @return string containing various configuration information about Gretl
 */
std::string about();

/**
 * @brief Returns a string for the Git SHA when the driver was built
 *
 * Note: This will not update unless you reconfigure CMake after a commit.
 *
 * @return string value of the Git SHA if built in a Git repo, empty if not
 */
std::string gitSHA();

/**
 * @brief Returns a string for the version of Gretl
 *
 * @param[in] add_SHA boolean for whether to add the Git SHA to the version if available
 *
 * @return string value of the version of Gretl
 */
std::string version(bool add_SHA = true);

/**
 * @brief Returns a string for the current compiler name and version
 *
 * @return string value of the current compiler name and version
 */
std::string compiler();

/**
 * @brief Returns a string for the current CMake build type (e.g. Debug, Release)
 *
 * @return string value of the build type
 */
std::string buildType();

}  // namespace gretl
