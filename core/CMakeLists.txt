cmake_minimum_required(VERSION 3.28)


project(RoughPy_Core
        VERSION 0.0.1
        LANGUAGES CXX
)

add_library(RoughPy_Core INTERFACE
        include/roughpy/core/alloc.h
        include/roughpy/core/hash.h
        include/roughpy/core/types.h
        include/roughpy/core/traits.h
        include/roughpy/core/macros.h
        include/roughpy/core/helpers.h
        include/roughpy/core/slice.h
)
add_library(RoughPy::Core ALIAS RoughPy_Core)


target_include_directories(RoughPy_Core INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(RoughPy_Core INTERFACE
        Boost::headers
)

#add_roughpy_component(Core INTERFACE
#        PUBLIC_HEADERS

#        DEPENDENCIES INTERFACE
#            Boost::boost
#        )

target_compile_definitions(RoughPy_Core INTERFACE NOMINMAX=1)


