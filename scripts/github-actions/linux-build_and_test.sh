#!/bin/bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and
# other Gretl Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
##############################################################################

set -x

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

echo "~~~~ helpful info ~~~~"
echo "USER="`id -u -n`
echo "PWD="`pwd`
echo "HOST_CONFIG=$HOST_CONFIG"
echo "CMAKE_EXTRA_FLAGS=$CMAKE_EXTRA_FLAGS"
echo "~~~~~~~~~~~~~~~~~~~~~~"

export BUILD_TYPE=${BUILD_TYPE:-Debug}


echo "~~~~~~ FIND NUMPROCS ~~~~~~~~"
NUMPROCS=`python3 -c "import os; print(f'{os.cpu_count()}')"`
NUM_BUILD_PROCS=`python3 -c "import os; print(f'{max(2, os.cpu_count() * 8 // 10)}')"`

echo "~~~~~~ RUNNING CMAKE ~~~~~~~~"
or_die python3 ./config-build.py -bp builddir -hc ./host-configs/docker/${HOST_CONFIG} -bt ${BUILD_TYPE} ${CMAKE_EXTRA_FLAGS}
or_die cd builddir

echo "~~~~~~ BUILDING ~~~~~~~~"
or_die make -j $NUM_BUILD_PROCS VERBOSE=1

echo "~~~~~~ RUNNING TESTS ~~~~~~~~"
make CTEST_OUTPUT_ON_FAILURE=1 test ARGS='-T Test -VV -j$NUM_BUILD_PROCS'

