cmake_minimum_required(VERSION 3.19)

project(RoughPy_Streams
        VERSION 0.0.1
        LANGUAGES CXX)


include(GNUInstallDirs)
include(GenerateExportHeader)

find_package(Boost 1.64 CONFIG REQUIRED COMPONENTS headers)


option(RPY_STREAMS_SHARED "Build shared RoughPy streams module" ON)

if (RPY_STREAMS_SHARED)
    add_library(RoughPy_Streams SHARED)
else()
    add_library(RoughPy_Streams STATIC)
endif()

add_library(RoughPy::Streams ALIAS RoughPy_Streams)


target_link_libraries(RoughPy_Streams PUBLIC
        RoughPy::Core
        RoughPy::Intervals
        RoughPy::Scalars
        RoughPy::Algebra
        Boost::boost
        )


set(ROUGHPY_STREAMS_PUBLIC_HEADERS
        include/roughpy/streams/dyadic_caching_layer.h
        include/roughpy/streams/dynamically_constructed_stream.h
        include/roughpy/streams/lie_increment_stream.h
        include/roughpy/streams/stream.h
        include/roughpy/streams/stream_base.h
        include/roughpy/streams/tick_stream.h
        include/roughpy/streams/brownian_stream.h
        )

target_sources(RoughPy_Streams
        PUBLIC
            ${ROUGHPY_STREAMS_PUBLIC_HEADERS}
        PRIVATE
            src/stream_base.cpp
            src/stream.cpp
            src/lie_increment_stream.cpp
            src/dynamically_constructed_stream.cpp
            src/dyadic_caching_layer.cpp
            src/piecewise_lie_stream.cpp
            src/tick_stream.cpp
            src/brownian_stream.cpp
        )


target_include_directories(RoughPy_Streams
        PRIVATE
            "include/roughpy/streams"
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )


generate_export_header(RoughPy_Streams)


add_roughpy_test(Streams
        SRC
            src/test_lie_increment_stream.cpp
            src/test_brownian_stream.cpp
        LINK_COMPONENT)
