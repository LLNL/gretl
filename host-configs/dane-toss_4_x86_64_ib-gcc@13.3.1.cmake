#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/tce/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@13.3.1/hpiynx6mdjb57v77us3h6yyzxymdbr6d
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/usr/WS2/smithdev/libs/smith/toss_4_x86_64_ib/2025_09_25_10_10_47/none-none/compiler-wrapper-1.0-hxzxhboumtniaqs7qivgliluv4h2lbxt/libexec/spack/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/WS2/smithdev/libs/smith/toss_4_x86_64_ib/2025_09_25_10_10_47/none-none/compiler-wrapper-1.0-hxzxhboumtniaqs7qivgliluv4h2lbxt/libexec/spack/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/WS2/smithdev/libs/smith/toss_4_x86_64_ib/2025_09_25_10_10_47/none-none/compiler-wrapper-1.0-hxzxhboumtniaqs7qivgliluv4h2lbxt/libexec/spack/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-13.3.1/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-13.3.1/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-13.3.1/bin/gfortran" CACHE PATH "")

endif()

set(CMAKE_Fortran_FLAGS "-fPIC" CACHE STRING "")


#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(DEVTOOLS_ROOT "/usr/WS2/smithdev/devtools/toss_4_x86_64_ib/2024_05_30_15_02_20/._view/psk2dcrijss6s4i6qmxplzthrzm3y7nh" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "/usr/tce/packages/clang/clang-19.1.3/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "/usr/tce/packages/clang/clang-19.1.3/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/python-3.11.7/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/cppcheck-2.9/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/doxygen-1.9.8/bin/doxygen" CACHE PATH "")


