cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)

add_roughpy_component(Platform
    SOURCES
        src/configuration.cpp
#        src/threading/openmp_threading.cpp
        src/errors.cpp
        src/fs_path_serialization.cpp
        src/devices/host/host_buffer.cpp
        src/devices/host/host_buffer.h
        src/devices/host/host_decls.h
        src/devices/host/host_device_impl.cpp
        src/devices/host/host_device_impl.h
        src/devices/host/host_device_provider.cpp
        src/devices/host/host_device_provider.h
        src/devices/host/host_event.cpp
        src/devices/host/host_event.h
        src/devices/host/host_kernel.cpp
        src/devices/host/host_kernel.h
        src/devices/host/host_queue.cpp
        src/devices/host/host_queue.h
        src/devices/opencl/ocl_buffer.cpp
        src/devices/opencl/ocl_buffer.h
        src/devices/opencl/ocl_decls.h
        src/devices/opencl/ocl_device.cpp
        src/devices/opencl/ocl_device.h
        src/devices/opencl/ocl_device_provider.cpp
        src/devices/opencl/ocl_device_provider.h
        src/devices/opencl/ocl_event.cpp
        src/devices/opencl/ocl_event.h
        src/devices/opencl/ocl_handle_errors.cpp
        src/devices/opencl/ocl_handle_errors.h
        src/devices/opencl/ocl_headers.h
        src/devices/opencl/ocl_helpers.cpp
        src/devices/opencl/ocl_helpers.h
        src/devices/opencl/ocl_kernel.cpp
        src/devices/opencl/ocl_kernel.h
        src/devices/opencl/ocl_queue.cpp
        src/devices/opencl/ocl_queue.h
        src/devices/opencl/ocl_version.cpp
        src/devices/opencl/ocl_version.h
        src/devices/buffer.cpp
        src/devices/buffer_interface.cpp
        src/devices/core.cpp
        src/devices/device_handle.cpp
        src/devices/device_interface_base.cpp
        src/devices/device_provider.cpp
        src/devices/event.cpp
        src/devices/event_interface.cpp
        src/devices/rational_numbers.cpp
        src/devices/get_device.cpp
        src/devices/kernel.cpp
        src/devices/kernel_arg.cpp
        src/devices/kernel_interface.cpp
        src/devices/kernel_launch_params.cpp
        src/devices/memory_view.cpp
        src/devices/queue.cpp
        src/devices/queue_interface.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/platform/devices.h
        include/roughpy/platform/errors.h
        include/roughpy/platform/devices/buffer.h
        include/roughpy/platform/devices/core.h
        include/roughpy/platform/devices/device_handle.h
        include/roughpy/platform/devices/device_provider.h
        include/roughpy/platform/devices/event.h
        include/roughpy/platform/devices/host_device.h
        include/roughpy/platform/devices/kernel.h
        include/roughpy/platform/devices/kernel_arg.h
        include/roughpy/platform/devices/macros.h
        include/roughpy/platform/devices/memory_view.h
        include/roughpy/platform/devices/queue.h
        include/roughpy/platform/devices/types.h
    DEPENDENCIES
        PUBLIC
            Boost::boost
            Boost::system
            Boost::url
            Boost::filesystem IF NOT RPY_HAS_STD_FILESYSTEM
            cereal::cereal
        Eigen3::Eigen
        Libalgebra_lite::Libalgebra_lite
        GMP::GMP
        PRIVATE
        OpenCL::OpenCL
    CONFIGURE
        FILE
            IN available_libraries.h.in
            OUT available_libraries.h
            ATONLY
    NEEDS
        RoughPy::Core
    )

target_precompile_headers(RoughPy_Platform PUBLIC
        <cereal/cereal.hpp>
        <cereal/access.hpp>
        <cereal/specialize.hpp>
        <cereal/types/base_class.hpp>
        <cereal/types/optional.hpp>
        <cereal/types/polymorphic.hpp>
        <cereal/types/string.hpp>
        <cereal/types/utility.hpp>
        <cereal/types/vector.hpp>
        <cereal/archives/binary.hpp>
        <cereal/archives/json.hpp>
        <cereal/archives/portable_binary.hpp>
        <cereal/archives/xml.hpp>
)

target_precompile_headers(RoughPy_Platform PUBLIC
        <boost/multiprecision/gmp.hpp>
        <libalgebra_lite/coefficients.h>
        <libalgebra_lite/polynomial.h>
)

if (APPLE)
    target_compile_definitions(RoughPy_Platform PUBLIC BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()


add_roughpy_test(Devices
        SRC
        src/devices/test_cpu_device.cpp
        src/devices/test_gpu_device.cpp
        DEP
        OpenCL::OpenCL
        NEEDS
        RoughPy::Platform
)
