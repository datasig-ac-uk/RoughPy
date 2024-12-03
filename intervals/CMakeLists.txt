

setup_roughpy_component(Intervals)

# TODO: I don't think this really needs to be a shared library
add_library(RoughPy_Intervals SHARED
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
        include/roughpy/intervals/interval.h
        include/roughpy/intervals/dyadic.h
        include/roughpy/intervals/dyadic_interval.h
        include/roughpy/intervals/real_interval.h
        include/roughpy/intervals/partition.h
        include/roughpy/intervals/segmentation.h
        src/intersection_and_union.cpp
)
add_library(RoughPy::Intervals ALIAS RoughPy_Intervals)

target_include_directories(RoughPy_Intervals PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/intervals
)

set_target_properties(RoughPy_Intervals PROPERTIES ROUGHPY_COMPONENT Intervals)

target_link_libraries(RoughPy_Intervals
        PUBLIC
            RoughPy::Platform
        PRIVATE
            RoughPy::PrecompiledHeaders)

# TODO: This should be removed, it's not sensible
target_compile_definitions(RoughPy_Intervals PRIVATE "RPY_COMPILING_INTERVALS")


set_library_version_properties(RoughPy_Intervals)

generate_export_header(RoughPy_Intervals)

add_subdirectory(src)
