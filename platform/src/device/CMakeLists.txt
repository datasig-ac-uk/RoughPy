
set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/device")

target_sources(RoughPy_Platform PUBLIC
        ${_header_dir}/memory.h
        ${_header_dir}/device_handle.h
        ${_header_dir}/host_address_memory.h

        PRIVATE
        device_handle.cpp
        host_address_memory.cpp
        host_device.cpp
        host_device.h
        memory.cpp
)