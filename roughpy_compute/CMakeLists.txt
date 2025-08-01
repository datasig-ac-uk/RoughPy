
cmake_minimum_required(VERSION 3.30)

project(RoughPy-Compute LANGUAGES CXX)


add_library(roughpy_compute INTERFACE)


target_include_directories(roughpy_compute INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DI}/..>
)