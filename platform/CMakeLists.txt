cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)


add_library(RoughPy_Device_Support INTERFACE
        include/roughpy/device_support/algorithms.h
        include/roughpy/device_support/fundamental_type.h
        include/roughpy/device_support/host_kernel.h
        include/roughpy/device_support/operators.h
        include/roughpy/device_support/algorithm_drivers/algorithm_base.h
        include/roughpy/device_support/algorithm_drivers/copy.h
        include/roughpy/device_support/algorithm_drivers/count.h
        include/roughpy/device_support/algorithm_drivers/fill.h
        include/roughpy/device_support/algorithm_drivers/find.h
        include/roughpy/device_support/algorithm_drivers/lexicographical_compare.h
        include/roughpy/device_support/algorithm_drivers/lower_bound.h
        include/roughpy/device_support/algorithm_drivers/max_element.h
        include/roughpy/device_support/algorithm_drivers/min_element.h
        include/roughpy/device_support/algorithm_drivers/mismatch.h
        include/roughpy/device_support/algorithm_drivers/reverse.h
        include/roughpy/device_support/algorithm_drivers/rotate.h
        include/roughpy/device_support/algorithm_drivers/shift_left.h
        include/roughpy/device_support/algorithm_drivers/shift_right.h
        include/roughpy/device_support/algorithm_drivers/swap_ranges.h
        include/roughpy/device_support/algorithm_drivers/upper_bound.h
)

target_include_directories(RoughPy_Device_Support INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


add_roughpy_component(Platform
        SOURCES
        src/alloc.cpp
        src/configuration.cpp
        src/fs_path_serialization.cpp
        src/devices/algorithm_drivers.cpp
        src/devices/algorithm_drivers.h
        src/devices/buffer.cpp
        src/devices/buffer_interface.cpp
        src/devices/core.cpp
        src/devices/device_handle.cpp
        src/devices/device_interface_base.cpp
        src/devices/event.cpp
        src/devices/event_interface.cpp
        src/devices/get_best_device.cpp
        src/devices/get_device.cpp
        src/devices/kernel.cpp
        src/devices/kernel_arg.cpp
        src/devices/kernel_interface.cpp
        src/devices/kernel_launch_params.cpp
        src/devices/queue.cpp
        src/devices/queue_interface.cpp
        src/devices/type.cpp
        src/devices/value.cpp
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
        src/devices/host/kernel_registry.h
        src/devices/fundamental_types/double_type.cpp
        src/devices/fundamental_types/double_type.h
        src/devices/fundamental_types/float_type.cpp
        src/devices/fundamental_types/float_type.h
        src/devices/fundamental_types/int8_type.cpp
        src/devices/fundamental_types/int8_type.h
        src/devices/fundamental_types/int16_type.cpp
        src/devices/fundamental_types/int16_type.h
        src/devices/fundamental_types/int32_type.cpp
        src/devices/fundamental_types/int32_type.h
        src/devices/fundamental_types/int64_type.cpp
        src/devices/fundamental_types/int64_type.h
        src/devices/fundamental_types/uint8_type.cpp
        src/devices/fundamental_types/uint8_type.h
        src/devices/fundamental_types/uint16_type.cpp
        src/devices/fundamental_types/uint16_type.h
        src/devices/fundamental_types/uint32_type.cpp
        src/devices/fundamental_types/uint32_type.h
        src/devices/fundamental_types/uint64_type.cpp
        src/devices/fundamental_types/uint64_type.h
        src/devices/algorithms/algorithm_impl.h
        src/devices/algorithms/algorithms_dispatcher.cpp
        src/devices/algorithms/copy.cpp
        src/devices/algorithms/count.cpp
        src/devices/algorithms/fill.cpp
        src/devices/algorithms/find.cpp
        src/devices/algorithms/lexicographical_compare.cpp
        src/devices/algorithms/lower_bound.cpp
        src/devices/algorithms/max.cpp
        src/devices/algorithms/min.cpp
        src/devices/algorithms/mismatch.cpp
        src/devices/algorithms/reverse.cpp
        src/devices/algorithms/shift_left.cpp
        src/devices/algorithms/shift_right.cpp
        src/devices/algorithms/swap.cpp
        src/devices/algorithms/upper_bound.cpp
        PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/alloc.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/devices/algorithms.h
        include/roughpy/devices/buffer.h
        include/roughpy/devices/core.h
        include/roughpy/devices/device_handle.h
        include/roughpy/devices/device_object_base.h
        include/roughpy/devices/device_provider.h
        include/roughpy/devices/event.h
        include/roughpy/devices/host_device.h
        include/roughpy/devices/kernel.h
        include/roughpy/devices/kernel_arg.h
        include/roughpy/devices/macros.h
        include/roughpy/devices/queue.h
        include/roughpy/devices/type.h
        include/roughpy/devices/type_dispatcher.h
        include/roughpy/devices/value.h
        DEPENDENCIES
        PUBLIC
        Boost::boost
        Boost::system
        Boost::url
        Boost::filesystem IF NOT RPY_HAS_STD_FILESYSTEM
        cereal::cereal
        PRIVATE
        RoughPy_Device_Support
        CONFIGURE
        FILE
        IN available_libraries.h.in
        OUT available_libraries.h
        ATONLY
        NEEDS
        RoughPy::Core
)
target_include_directories(RoughPy_Platform PRIVATE
        include/roughpy
        include/roughpy/devices
        src
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


if (APPLE)
    target_compile_definitions(RoughPy_Platform PUBLIC BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif ()

add_roughpy_test(Devices SRC
        src/devices/test_value.cpp
        NEEDS
        RoughPy::Platform)
