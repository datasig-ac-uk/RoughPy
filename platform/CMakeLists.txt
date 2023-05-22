cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)

add_roughpy_lib(Platform STATIC
    SOURCES
        src/configuration.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
    PUBLIC_DEPS
        RoughPy::Core
        Boost::boost
        Boost::url
        Boost::disable_autolinking
        cereal::cereal
    )
