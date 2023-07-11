cmake_minimum_required(VERSION 3.21)

project(RoughPy_Intervals
        VERSION 0.0.1
        LANGUAGES CXX)


include(GNUInstallDirs)
include(GenerateExportHeader)


add_roughpy_component(Intervals
        SOURCES
        src/interval.cpp
        src/dyadic.cpp
        src/dyadic_interval.cpp
        src/real_interval.cpp
        src/segmentation.cpp
        src/scaled_predicate.cpp
        src/scaled_predicate.h
        src/dyadic_searcher.cpp
        src/dyadic_searcher.h
        src/partition.cpp
        PUBLIC_HEADERS
        include/roughpy/intervals/interval.h
        include/roughpy/intervals/dyadic.h
        include/roughpy/intervals/dyadic_interval.h
        include/roughpy/intervals/real_interval.h
        include/roughpy/intervals/partition.h
        include/roughpy/intervals/segmentation.h
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        )


add_roughpy_test(Intervals
        SRC
        src/test_dyadic.cpp
        src/test_dyadic_intervals.cpp
        src/test_real_interval.cpp
        src/test_partition.cpp
        NEEDS
        RoughPy::Intervals
        )
