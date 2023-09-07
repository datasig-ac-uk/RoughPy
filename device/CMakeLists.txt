cmake_minimum_required(VERSION 3.21)


project(RoughPy_Device VERSION 0.0.1
        LANGUAGES CXX)


add_roughpy_component(Device
        PUBLIC_HEADERS
        include/roughpy/device/buffer.h
        include/roughpy/device/core.h
        include/roughpy/device/macros.h
        include/roughpy/device/device_handle.h
        include/roughpy/device/device_object_base.h
        include/roughpy/device/event.h
        include/roughpy/device/kernel.h
        include/roughpy/device/macros.h
        include/roughpy/device/queue.h
        SOURCES
        src/opencl/open_cl_device.cpp
        src/opencl/open_cl_device.h
        src/opencl/ocl_buffer.cpp
        src/opencl/ocl_buffer.h
        src/opencl/ocl_event.cpp
        src/opencl/ocl_event.h
        src/opencl/ocl_queue.cpp
        src/opencl/ocl_queue.h
    src/opencl/ocl_kernel.cpp
    src/opencl/ocl_kernel.h
        src/opencl/opencl_headers.h
        src/buffer.cpp
        src/device.cpp
        src/device_object_base.cpp
        src/event.cpp
        src/get_device.cpp
        src/kernel.cpp
        src/queue.cpp
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        PRIVATE_DEPS
        OpenCL::OpenCL
        )
