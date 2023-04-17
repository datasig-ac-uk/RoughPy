cmake_minimum_required(VERSION 3.19)

project(RoughPy_Core
        VERSION 0.0.1
        LANGUAGES CXX
        )

include(GNUInstallDirs)

add_roughpy_lib(Core INTERFACE
    PUBLIC_HEADERS
        include/roughpy/core/implementation_types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h
        )
