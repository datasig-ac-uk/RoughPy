

setup_roughpy_component(Core)

add_library(RoughPy_Core STATIC
        include/roughpy/core/detail/config.h
        include/roughpy/core/detail/bit_cast.hpp
        include/roughpy/core/construct_inplace.hpp
        include/roughpy/core/check.h
        include/roughpy/core/check_helpers.hpp
        include/roughpy/core/debug_assertion.h
        include/roughpy/core/fnv1a_hash.h
        include/roughpy/core/hash.hpp
        include/roughpy/core/types.hpp
        include/roughpy/core/traits.hpp
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.hpp
        include/roughpy/core/slice.hpp
        include/roughpy/core/smart_ptr.hpp
        include/roughpy/core/string_utils.hpp
        include/roughpy/core/ranges.hpp
        include/roughpy/core/meta.hpp
        src/fnv1a_hash.c
)
add_library(RoughPy::Core ALIAS RoughPy_Core)


target_include_directories(RoughPy_Core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(RoughPy_Core INTERFACE
        Boost::headers
        range-v3::range-v3
)

set_target_properties(RoughPy_Core PROPERTIES
        ROUGHPY_COMPONENT Core
        POSITION_INDEPENDENT_CODE ON
)

#add_roughpy_component(Core INTERFACE
#        PUBLIC_HEADERS

#        DEPENDENCIES INTERFACE
#            Boost::boost
#        )

target_compile_definitions(RoughPy_Core INTERFACE NOMINMAX=1)


add_subdirectory(src)