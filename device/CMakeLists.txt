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
        include/roughpy/device/host_device.h
        include/roughpy/device/kernel.h
        include/roughpy/device/kernel_arg.h
        include/roughpy/device/macros.h
        include/roughpy/device/queue.h
        include/roughpy/device/types.h
        include/roughpy/device.h
        SOURCES
#        src/cpu/kernels/masked_binary.h
#        src/cpu/kernels/masked_binary_bfloat16.cpp
#        src/cpu/kernels/masked_binary_double.cpp
#        src/cpu/kernels/masked_binary_float.cpp
#        src/cpu/kernels/masked_binary_half.cpp
#        src/cpu/kernels/masked_binary_poly.cpp
#        src/cpu/kernels/masked_binary_rational.cpp
#        src/cpu/kernels/masked_unary.h
#        src/cpu/kernels/masked_unary_bfloat16.cpp
#        src/cpu/kernels/masked_unary_double.cpp
#        src/cpu/kernels/masked_unary_float.cpp
#        src/cpu/kernels/masked_unary_half.cpp
#        src/cpu/kernels/masked_unary_poly.cpp
#        src/cpu/kernels/masked_unary_rational.cpp
#        src/cpu/kernels/operators.h
        src/cpu/cpu_buffer.cpp
        src/cpu/cpu_buffer.h
        src/cpu/cpu_decls.h
        src/cpu/cpu_device.cpp
        src/cpu/cpu_device.h
        src/cpu/cpu_device_provider.cpp
        src/cpu/cpu_device_provider.h
        src/cpu/cpu_event.cpp
        src/cpu/cpu_event.h
        src/cpu/cpu_kernel.cpp
        src/cpu/cpu_kernel.h
        src/cpu/cpu_queue.cpp
        src/cpu/cpu_queue.h
        src/opencl/ocl_buffer.cpp
        src/opencl/ocl_buffer.h
        src/opencl/ocl_decls.h
        src/opencl/ocl_device.cpp
        src/opencl/ocl_device.h
        src/opencl/ocl_device_provider.cpp
        src/opencl/ocl_device_provider.h
        src/opencl/ocl_event.cpp
        src/opencl/ocl_event.h
        src/opencl/ocl_handle_errors.cpp
        src/opencl/ocl_handle_errors.h
        src/opencl/ocl_headers.h
        src/opencl/ocl_helpers.cpp
        src/opencl/ocl_helpers.h
        src/opencl/ocl_kernel.cpp
        src/opencl/ocl_kernel.h
        src/opencl/ocl_queue.cpp
        src/opencl/ocl_queue.h
        src/opencl/ocl_version.cpp
        src/opencl/ocl_version.h
        src/buffer.cpp
        src/buffer_interface.cpp
        src/device_handle.cpp
        src/device_interface_base.cpp
        src/device_provider.cpp
        src/event.cpp
        src/event_interface.cpp
        src/get_device.cpp
        src/kernel.cpp
        src/kernel_arg.cpp
        src/kernel_interface.cpp
        src/kernel_launch_params.cpp
        src/queue.cpp
        src/queue_interface.cpp
        DEPENDENCIES
        PRIVATE
        OpenCL::OpenCL
        PUBLIC
            Eigen3::Eigen
            Libalgebra_lite::Libalgebra_lite
        NEEDS
        RoughPy::Core
        RoughPy::Platform
)



add_roughpy_test(Devices
    SRC
    src/opencl/test_ocl_version.cpp
    src/test_cpu_device.cpp
    src/test_gpu_device.cpp
    DEP
        OpenCL::OpenCL
    NEEDS
    RoughPy::Device
)
