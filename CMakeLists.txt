cmake_minimum_required(VERSION 3.18)
project(RoughPy VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(ROUGHPY_BUILD_TESTS "Build C++ tests for RoughPy" ON)

if (ROUGHPY_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)

    if(NOT TARGET GTest::gtest)
        message(FATAL_ERROR "GTest::gtest target not defined")
    endif()

    enable_testing()
endif()

include(cmake/roughpy_test_helpers.cmake)


add_subdirectory(external/libalgebra_lite)
add_subdirectory(external/pybind11)
# DLPACK has an annoying feature where it tries to build a demo
# library. Disable this here
set(BUILD_MOCK OFF CACHE BOOL INTERNAL)
add_subdirectory(external/dlpack)

add_subdirectory(config)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(streams)
add_subdirectory(roughpy)
