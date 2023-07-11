cmake_minimum_required(VERSION 3.21)


project(RoughPy_Device VERSION 0.0.1
        LANGUAGES CXX CUDA)


add_roughpy_lib(Device SHARED
        PUBLIC_HEADERS
        include/roughpy/device/core.h
        include/roughpy/device/device_context.h
        include/roughpy/device/device_algebra_base.h
        SOURCES
        kernels/functors.h
        kernels/unary_kernel.cuh
        kernels/binary_kernel.cuh
        kernels/tmp.cu
        src/device_context.cpp
        src/device_algebra_base.cpp
        src/device_free_tensor.cpp
        src/device_free_tensor.h
        PUBLIC_DEPS
        RoughPy::Core
        RoughPy::Scalars
        RoughPy::Algebra
        PVT_INCLUDE_DIRS
        kernels
        )


add_subdirectory(cuda)
