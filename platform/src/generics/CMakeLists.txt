

set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        ${_header_dir}/type.h
        ${_header_dir}/values.h

        into_from.cpp
        into_from.h
        type.cpp)