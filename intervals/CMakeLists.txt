cmake_minimum_required(VERSION 3.19)

project(RoughPy_Intervals
        VERSION 0.0.1
        LANGUAGES CXX)


include(GNUInstallDirs)
include(GenerateExportHeader)

find_package(Boost 1.64 CONFIG REQUIRED COMPONENTS headers)

option(RPY_INTERVALS_SHARED "Build shared RoughPy intervals module" ON)


if (RPY_INTERVALS_SHARED)
    add_library(RoughPy_Intervals SHARED)
else()
    add_library(RoughPy_Intervals STATIC)
endif()

add_library(RoughPy::Intervals ALIAS RoughPy_Intervals)

set(ROUGHPY_INTERVALS_PUBLIC_HEADERS
        include/roughpy/intervals/interval.h
        include/roughpy/intervals/dyadic.h
        include/roughpy/intervals/dyadic_interval.h
        include/roughpy/intervals/real_interval.h
        include/roughpy/intervals/segmentation.h
        )


target_sources(RoughPy_Intervals
        PUBLIC
            ${ROUGHPY_INTERVALS_PUBLIC_HEADERS}
        PRIVATE
            src/interval.cpp
            src/dyadic.cpp
            src/dyadic_interval.cpp
            src/real_interval.cpp
            src/segmentation.cpp
            src/scaled_predicate.cpp
            src/scaled_predicate.h
            src/dyadic_searcher.cpp
            src/dyadic_searcher.h
)


target_include_directories(RoughPy_Intervals
        PRIVATE
            "include/roughpy/intervals"
            "${CMAKE_CURRENT_BINARY_DIR}"
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )


generate_export_header(RoughPy_Intervals)



add_roughpy_test(Intervals
        SRC
            src/test_dyadic.cpp
            src/test_dyadic_intervals.cpp
        LINK_COMPONENT)
