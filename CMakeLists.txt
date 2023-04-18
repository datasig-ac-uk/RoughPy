cmake_minimum_required(VERSION 3.18)
project(RoughPy VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ROUGHPY_BUILD_LA_CONTEXTS "Build the collection of libalgebra contexts" OFF)
option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)

if (ROUGHPY_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)

    if(NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif()

    enable_testing()
endif()

include(cmake/roughpy_helpers.cmake)


option(ROUGHPY_LINK_NUMPY "Link with Numpy library for array handling" ON)

set(PYBIND11_FINDPYTHON ON)
if (NOT PYTHON_FOUND AND SKBUILD)
    cmake_path(GET "${PYTHON_EXECUTABLE}" PARENT_PATH _sk_env_dir)
    message(STATUS "SKBuild environment: ${_sk_env_dir}")

    # Some variables that are set might cause issues in the FindPython module,
    # so unset those
    unset(Python_LIBRARY CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(Python3_LIBRARY CACHE)

    # clean up temporary
    unset(_sk_env_dir)
else ()
    set(Python_FIND_VIRTUALENV FIRST)
endif ()

if (ROUGHPY_LINK_NUMPY)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
else ()
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
endif ()

if (DEFINED ENV{VIRTUAL_ENV})
    # Put venv/lib on the prefix path so we can find
    # a pip installed MKL
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}/lib")
endif()


find_package(Boost 1.81 REQUIRED COMPONENTS filesystem url)
find_package(Eigen3 CONFIG REQUIRED)
find_package(SndFile CONFIG REQUIRED)
find_package(CURL REQUIRED)


# This package is c++17 so cannot be used if changing to a lower standard
find_package(tomlplusplus CONFIG REQUIRED)

find_package(MKL CONFIG QUIET)


add_subdirectory(external/libalgebra_lite)
add_subdirectory(external/pybind11)
# DLPACK has an annoying feature where it tries to build a demo
# library. Disable this here
set(BUILD_MOCK OFF CACHE BOOL INTERNAL)
add_subdirectory(external/dlpack)

# Make an imported target for PCG-CPP because it is a
# makefile based project
if(NOT TARGET PCGRandom::pcg_random)
    add_library(PCGRandom::pcg_random IMPORTED INTERFACE)
    target_include_directories(PCGRandom::pcg_random INTERFACE
            external/pcg-cpp/include)
endif()

add_subdirectory(external/recombine)
add_subdirectory(external/libalgebra)

# The Cmake setup for rapidcsv leaves much to be desired.
# instead I'm going to define the target myself
if (NOT TARGET RapidCSV::rapidcsv)
    add_library(RapidCSV::rapidcsv IMPORTED INTERFACE)
    target_include_directories(RapidCSV::rapidcsv INTERFACE
            external/rapidcsv/src)
endif()



add_subdirectory(core)
add_subdirectory(platform)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)
add_subdirectory(roughpy)

if(ROUGHPY_BUILD_LA_CONTEXTS)
    add_subdirectory(la_context)
endif()
