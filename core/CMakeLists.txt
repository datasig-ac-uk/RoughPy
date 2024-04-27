cmake_minimum_required(VERSION 3.21)

project(RoughPy_Core
        VERSION 0.0.1
        LANGUAGES CXX
)

include(GNUInstallDirs)

add_roughpy_component(Core INTERFACE
        PUBLIC_HEADERS
        include/roughpy/core/container/map.h
        include/roughpy/core/container/unordered_map.h
        include/roughpy/core/container/unordered_set.h
        include/roughpy/core/container/vector.h
        include/roughpy/core/alloc.h
        include/roughpy/core/errors.h
        include/roughpy/core/hash.h
        include/roughpy/core/types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/ranges.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h
        include/roughpy/core/smart_ptr.h
        DEPENDENCIES INTERFACE
        Boost::boost
        range-v3::range-v3
)

target_compile_definitions(RoughPy_Core INTERFACE NOMINMAX=1)


if (MSVC)
    target_compile_options(RoughPy_Core INTERFACE /permissive-)
endif ()
