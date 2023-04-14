cmake_minimum_required(VERSION 3.19)


project(Roughpy_Platform VERSION 0.0.1)

add_roughpy_lib(Platform
    SOURCES
        src/tmp.cpp
    PUBLIC_HEADERS
        include/roughpy/platform/filesystem.h
    PUBLIC_DEPS
        Boost::boost
        Boost::filesystem
    )
