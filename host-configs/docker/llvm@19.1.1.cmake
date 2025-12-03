#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm@19.1.1/yq5rjyb4vplepvexmd5okbdyoezvrvsz
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/lib/llvm-19/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/lib/llvm-19/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-13" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_CXX_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_Fortran_FLAGS "-fPIC -pthread" CACHE STRING "")


#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

set(DEVTOOLS_ROOT "/usr" CACHE PATH "")

set(CLANGFORMAT_EXECUTABLE "${DEVTOOLS_ROOT}/lib/llvm-19/bin/clang-format" CACHE PATH "")

set(CLANGTIDY_EXECUTABLE "${DEVTOOLS_ROOT}/lib/llvm-19/bin/clang-tidy" CACHE PATH "")

set(ENABLE_DOCS ON CACHE BOOL "")

set(SPHINX_EXECUTABLE "${DEVTOOLS_ROOT}/bin/sphinx-build" CACHE PATH "")

set(CPPCHECK_EXECUTABLE "${DEVTOOLS_ROOT}/bin/cppcheck" CACHE PATH "")

set(DOXYGEN_EXECUTABLE "${DEVTOOLS_ROOT}/local/bin/doxygen" CACHE PATH "")


