cmake_minimum_required(VERSION 3.19)

project(RoughPy_Core
        VERSION 0.0.1
        LANGUAGES CXX
        )

include(GNUInstallDirs)

add_library(RoughPy_Core INTERFACE)
add_library(RoughPy::Core ALIAS RoughPy_Core)

target_sources(RoughPy_Core INTERFACE
        include/roughpy/core/implementation_types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h)


target_include_directories(RoughPy_Core INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )
