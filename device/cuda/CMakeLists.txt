cmake_minimum_required(VERSION 3.21)


project(RoughPy_Cuda
        VERSION 0.0.0
        LANGUAGES CXX CUDA
        )


find_package(Thrust CONFIG REQUIRED)

add_library(RoughPy_CUDA SHARED)


generate_export_header(RoughPy_CUDA)

set_target_properties(RoughPy_CUDA PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARATE_COMPILATION ON)

target_sources(RoughPy_CUDA PRIVATE
        include/roughpy/cuda/cuda_core.cuh)


add_subdirectory(scalars)

target_include_directories(RoughPy_CUDA PRIVATE
        include/roughpy/cuda
        ${CMAKE_CURRENT_BINARY_DIR})


target_link_libraries(RoughPy_CUDA PRIVATE
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Algebra
        Thrust::Thrust)
