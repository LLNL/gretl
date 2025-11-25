// Copyright (c) Lawrence Livermore National Security, LLC and
// other Gretl Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gretl/about.hpp"
#include "gretl/config.hpp"
#include "gretl/git_sha.hpp"

#include <string>

namespace gretl {

std::string about()
{
  std::string about = "\n";

  // Version info
  about += std::string("Gretl Version:    ") + version() + "\n";
  about += "\n";

  // General configuration
#ifdef GRETL_DEBUG
  about += std::string("Debug Build:      ON\n");
#else
  about += std::string("Debug Build:      OFF\n");
#endif

  about += std::string("\n");

  return about;
}

std::string gitSHA() { return GRETL_GIT_SHA; }

std::string version(bool add_SHA)
{
  std::string version = GRETL_VERSION_FULL;

  std::string sha = gitSHA();
  if (add_SHA && !sha.empty()) {
    version += "-" + sha;
  }

  return version;
}

std::string compiler() { return std::string(GRETL_COMPILER_NAME) + " version " + std::string(GRETL_COMPILER_VERSION); }

std::string buildType()
{
#ifdef GRETL_DEBUG
  return "Debug";
#else
  return "Release";
#endif
}

}  // namespace gretl
