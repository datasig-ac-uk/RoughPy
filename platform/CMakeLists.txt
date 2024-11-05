cmake_minimum_required(VERSION 3.21)


project(Roughpy_Platform VERSION 0.0.1)


set_component_path_variables(Platform)

add_Library(RoughPy_Platform SHARED
        src/configuration.cpp
        #        src/threading/openmp_threading.cpp
        src/errors.cpp
        src/fs_path_serialization.cpp

        include/roughpy/platform.h
        include/roughpy/platform/filesystem.h
        include/roughpy/platform/configuration.h
        include/roughpy/platform/serialization.h
        include/roughpy/platform/threads.h
        include/roughpy/platform/devices.h
        include/roughpy/platform/errors.h

)
add_library(RoughPy::Platform ALIAS RoughPy_Platform)

target_include_directories(RoughPy_Platform PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/platform
)

target_link_libraries(RoughPy_Platform PUBLIC
        RoughPy::Core
        Boost::system
        Boost::url
        cereal::cereal
        Eigen3::Eigen
        GMP::GMP
        Libalgebra_lite::Libalgebra_lite
        PRIVATE
        OpenCL::OpenCL
)


add_subdirectory(src/devices)


#add_roughpy_component(Platform
#    SOURCES

#    PUBLIC_HEADERS

#    DEPENDENCIES
#        PUBLIC
#            Boost::boost
#            Boost::system
#            Boost::url
#            Boost::filesystem IF NOT RPY_HAS_STD_FILESYSTEM
#            cereal::cereal
#        Eigen3::Eigen
#        Libalgebra_lite::Libalgebra_lite
#        GMP::GMP
#        PRIVATE
#        OpenCL::OpenCL
#    CONFIGURE
#        FILE
#            IN available_libraries.h.in
#            OUT available_libraries.h
#            ATONLY
#    NEEDS
#        RoughPy::Core
#    )

target_precompile_headers(RoughPy_Platform PUBLIC
        <cereal/cereal.hpp>
        <cereal/access.hpp>
        <cereal/specialize.hpp>
        <cereal/types/base_class.hpp>
        <cereal/types/optional.hpp>
        <cereal/types/polymorphic.hpp>
        <cereal/types/string.hpp>
        <cereal/types/utility.hpp>
        <cereal/types/vector.hpp>
        <cereal/archives/binary.hpp>
        <cereal/archives/json.hpp>
        <cereal/archives/portable_binary.hpp>
        <cereal/archives/xml.hpp>
)

target_precompile_headers(RoughPy_Platform PUBLIC
        <boost/multiprecision/gmp.hpp>
        <libalgebra_lite/coefficients.h>
        <libalgebra_lite/polynomial.h>
)

#
#add_roughpy_test(Devices
#        SRC
#        src/devices/test_cpu_device.cpp
#        src/devices/test_gpu_device.cpp
#        DEP
#        OpenCL::OpenCL
#        NEEDS
#        RoughPy::Platform
#)
