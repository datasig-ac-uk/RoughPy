

set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        ${_header_dir}/arithmetic_trait.h
        ${_header_dir}/builtin_trait.h
        ${_header_dir}/comparison_trait.h
        ${_header_dir}/hash_trait.h
        ${_header_dir}/number_trait.h
        ${_header_dir}/type.h
        ${_header_dir}/values.h

        arithmetic_trait.cpp
        comparison_trait.cpp
        hash_trait.cpp
        into_from.cpp
        into_from.h
        type.cpp
        value.cpp
)