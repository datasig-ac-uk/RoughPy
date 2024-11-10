
set(_generics_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        "${_generics_header_dir}/builtin_traits/hashable.h"
        "${_generics_header_dir}/traits_detail/trait_base.h"
        "${_generics_header_dir}/const_reference.h"
        "${_generics_header_dir}/reference.h"
        "${_generics_header_dir}/traits.h"
        "${_generics_header_dir}/value.h"

        "arithmetic_trait.cpp"
        "equality_trait.cpp"
        "number_traits.cpp"
        "ordering_trait.cpp"

)