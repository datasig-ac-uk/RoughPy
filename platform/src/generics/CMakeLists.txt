

set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        ${_header_dir}/arithmetic_trait.h
        ${_header_dir}/builtin_trait.h
        ${_header_dir}/comparison_trait.h
        ${_header_dir}/conversion_trait.h
        ${_header_dir}/hash_trait.h
        ${_header_dir}/number_trait.h
        ${_header_dir}/type.h
        ${_header_dir}/values.h

        arithmetic_trait.cpp
        backup_display.cpp
        comparison_trait.cpp
        conversion_trait.cpp
        hash_trait.cpp
        number_trait.cpp
        type.cpp
        value.cpp
)