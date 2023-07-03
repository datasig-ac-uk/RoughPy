cmake_minimum_required(VERSION 3.21)

project(RoughPy_Core
        VERSION 0.0.1
        LANGUAGES CXX
        )

include(GNUInstallDirs)

add_roughpy_component(Core INTERFACE
        PUBLIC_HEADERS
        include/roughpy/core/types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h
        include/roughpy/core/alloc.h
        DEFINITIONS
        RPY_DBG_ASSERT_USE_EXCEPTIONS 1
        )
