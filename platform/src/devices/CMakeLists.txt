



#TODO: Handle with care, ideally device implementations should be static libs
#     linked into the main lib

target_sources(RoughPy_Platform PRIVATE
        buffer.cpp
        buffer_interface.cpp
        core.cpp
        device_handle.cpp
        device_interface_base.cpp
        device_provider.cpp
        event.cpp
        event_interface.cpp
        rational_numbers.cpp
        get_device.cpp
        kernel.cpp
        kernel_arg.cpp
        kernel_interface.cpp
        kernel_launch_params.cpp
        memory_view.cpp
        queue.cpp
        queue_interface.cpp
        PUBLIC
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/buffer.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/core.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/device_handle.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/device_provider.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/event.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/host_device.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/kernel.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/kernel_arg.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/macros.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/memory_view.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/queue.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/platform/devices/types.h
)

add_subdirectory(host)
add_subdirectory(opencl)