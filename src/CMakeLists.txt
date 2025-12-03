# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(gretl)

if(GRETL_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
