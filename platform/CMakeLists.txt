cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)






add_roughpy_component(Platform
    SOURCES
        src/configuration.cpp
#        src/threading/openmp_threading.cpp
        src/device.cpp
        src/cpu_device/CPUDevice.cpp
        src/cpu_device/CPUDevice.h
        src/fs_path_serialization.cpp
    PUBLIC_HEADERS
        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/platform/device.h
    DEPENDENCIES
        PUBLIC
            Boost::boost
            Boost::system
            Boost::url
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
