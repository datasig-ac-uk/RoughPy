



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
        INTERFACE
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/buffer.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/core.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/device_handle.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/device_provider.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/event.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/host_device.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/kernel.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/kernel_arg.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/macros.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/memory_view.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/queue.h
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices/types.h
)

add_subdirectory(host)
add_subdirectory(opencl)