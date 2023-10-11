cmake_minimum_required(VERSION 3.21)


project(RoughPy_Device VERSION 0.0.1
        LANGUAGES CXX)


add_roughpy_component(Device
        PUBLIC_HEADERS
        include/roughpy/device/buffer.h
        include/roughpy/device/core.h
        include/roughpy/device/device_handle.h
        include/roughpy/device/device_object_base.h
        include/roughpy/device/event.h
        include/roughpy/device/kernel.h
        include/roughpy/device/macros.h
        include/roughpy/device/queue.h
        SOURCES
        src/opencl/ocl_buffer.cpp
        src/opencl/ocl_buffer.h
        src/opencl/ocl_device.cpp
        src/opencl/ocl_device.h
        src/opencl/ocl_event.cpp
        src/opencl/ocl_event.h
        src/opencl/ocl_handle_errors.cpp
        src/opencl/ocl_handle_errors.h
        src/opencl/ocl_headers.h
        src/opencl/ocl_kernel.cpp
        src/opencl/ocl_kernel.h
        src/opencl/ocl_queue.cpp
        src/opencl/ocl_queue.h
        src/buffer.cpp
        src/buffer_interface.cpp
        src/device_handle.cpp
        src/device_interface_base.cpp
        src/event.cpp
        src/event_interface.cpp
        src/get_device.cpp
        src/kernel.cpp
        src/kernel_interface.cpp
        src/kernel_launch_params.cpp
        src/queue.cpp
        src/queue_interface.cpp
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        )
