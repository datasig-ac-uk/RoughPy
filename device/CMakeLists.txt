cmake_minimum_required(VERSION 3.21)


project(RoughPy_Device VERSION 0.0.1
        LANGUAGES CXX)


add_roughpy_component(Device
        PUBLIC_HEADERS
        include/roughpy/device/core.h
        include/roughpy/device/macros.h
        include/roughpy/device/device_handle.h
        include/roughpy/device/kernel.h
        SOURCES
        src/device.cpp
        src/opencl/open_cl_runtime_library.cpp
        src/opencl/open_cl_runtime_library.h
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        PRIVATE_DEPS
        OpenCL::OpenCL
        )
