
set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/devices")

target_sources(RoughPy_Platform PUBLIC
        ${_header_dir}/memory.h

        PRIVATE
        device_handle.cpp
        memory.cpp
)