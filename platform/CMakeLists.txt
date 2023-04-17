cmake_minimum_required(VERSION 3.19)


project(Roughpy_Platform VERSION 0.0.1)

add_roughpy_lib(Platform STATIC
    SOURCES
        src/tmp.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
    PUBLIC_DEPS
        RoughPy::Core
        Boost::boost
        Boost::filesystem
        Boost::url
        Boost::disable_autolinking
    )
