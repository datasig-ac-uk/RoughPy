

setup_roughpy_component(Core)

add_library(RoughPy_Core INTERFACE
        include/roughpy/core/detail/config.h
        include/roughpy/core/detail/bit_cast.h
        include/roughpy/core/construct_inplace.h
        include/roughpy/core/check.h
        include/roughpy/core/check_helpers.h
        include/roughpy/core/debug_assertion.h
        include/roughpy/core/hash.h
        include/roughpy/core/types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h
        include/roughpy/core/smart_ptr.h
        include/roughpy/core/string_utils.h
        include/roughpy/core/ranges.h
        include/roughpy/core/meta.h
)
add_library(RoughPy::Core ALIAS RoughPy_Core)


target_include_directories(RoughPy_Core INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(RoughPy_Core INTERFACE
        Boost::headers
        range-v3::range-v3
)

set_target_properties(RoughPy_Core PROPERTIES ROUGHPY_COMPONENT Core)

#add_roughpy_component(Core INTERFACE
#        PUBLIC_HEADERS

#        DEPENDENCIES INTERFACE
#            Boost::boost
#        )

target_compile_definitions(RoughPy_Core INTERFACE NOMINMAX=1)


add_subdirectory(src)