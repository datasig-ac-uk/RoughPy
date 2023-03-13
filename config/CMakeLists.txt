cmake_minimum_required(VERSION 3.19)

project(RoughPy_Config
        VERSION 0.0.1
        LANGUAGES CXX
        )

include(GNUInstallDirs)

add_library(RoughPy_Config INTERFACE)
add_library(RoughPy::Config ALIAS RoughPy_Config)

target_sources(RoughPy_Config INTERFACE
        include/roughpy/config/implementation_types.h
        include/roughpy/config/traits.h
        include/roughpy/config/macros.h
        include/roughpy/config/helpers.h)


target_include_directories(RoughPy_Config INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )
