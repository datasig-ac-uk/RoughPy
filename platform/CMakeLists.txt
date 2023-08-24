cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)

include(CheckIncludeFileCXX)

check_include_file_cxx(filesystem RPY_HAS_STD_FILESYSTEM)




add_roughpy_component(Platform
    SOURCES
        src/configuration.cpp
#        src/threading/openmp_threading.cpp
        src/runtime_library.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/platform/runtime_library.h
    PUBLIC_DEPS
        Boost::boost
        Boost::url
        Boost::disable_autolinking
        cereal::cereal
    CONFIGURE
        FILE
            IN available_libraries.h.in
            OUT available_libraries.h
            ATONLY
    NEEDS
        RoughPy::Core
    )
