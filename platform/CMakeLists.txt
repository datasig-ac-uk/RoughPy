cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)






add_roughpy_component(Platform
    SOURCES
        src/configuration.cpp
        src/runtime_library.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/platform/runtime_library.h
    DEPENDENCIES
        PUBLIC
            Boost::boost
            Boost::url
            Boost::system
            Boost::disable_autolinking
            Boost::filesystem IF NOT RPY_HAS_STD_FILESYSTEM
            cereal::cereal
    CONFIGURE
        FILE
            IN available_libraries.h.in
            OUT available_libraries.h
            ATONLY
    NEEDS
        RoughPy::Core
    )
