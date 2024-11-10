
set(_generics_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        "${_generics_header_dir}/builtin_traits/hashable.h"
        "${_generics_header_dir}/traits/arithmetic_trait.h"
        "${_generics_header_dir}/traits/builtin_trait.h"
        "${_generics_header_dir}/traits/dynamic_trait.h"
        "${_generics_header_dir}/traits/equality_trait.h"
        "${_generics_header_dir}/traits/hashable_trait.h"
        "${_generics_header_dir}/traits/number_trait.h"
        "${_generics_header_dir}/traits/ordering_trait.h"
        "${_generics_header_dir}/traits/static_trait.h"
        "${_generics_header_dir}/traits/trait.h"
        "${_generics_header_dir}/traits/trait_casts.h"
        "${_generics_header_dir}/const_reference.h"
        "${_generics_header_dir}/reference.h"
        "${_generics_header_dir}/type.h"
        "${_generics_header_dir}/type_ptr.h"
        "${_generics_header_dir}/value.h"
        "arithmetic_trait.cpp"
        "equality_trait.cpp"
        "number_traits.cpp"
        "ordering_trait.cpp"
        "type.cpp"
)


