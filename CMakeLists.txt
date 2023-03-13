cmake_minimum_required(VERSION 3.18)
project(RoughPy VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(external/libalgebra_lite)
add_subdirectory(external/pybind11)

add_subdirectory(config)
add_subdirectory(scalars)
add_subdirectory(intervals)
add_subdirectory(algebra)
add_subdirectory(paths)
add_subdirectory(roughpy)
