

setup_roughpy_component(Platform)

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
        src/generics/type.cpp


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

set_target_properties(RoughPy_Platform PROPERTIES ROUGHPY_COMPONENT Platform)

if (APPLE)
    target_compile_definitions(RoughPy_Platform PUBLIC BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
endif()


set_library_version_properties(RoughPy_Platform)


# TODO: This should be removed, it's not sensible
target_compile_definitions(RoughPy_Platform PRIVATE "RPY_COMPILING_PLATFORM")


generate_export_header(RoughPy_Platform
        EXPORT_FILE_NAME roughpy/platform/roughpy_platform_export.h
    )

add_subdirectory(src)


# Cereal is our current serialization library, and it's dreadful.
# Parsing these headers and instantiating all the templates therein is
# easily the biggest contribution to the build type for RoughPy. For this
# reason, we precompile all the headers for cereal. Ideally, I want a less
# intensive solution like protobufs or flatbufs, or literally anything else.
if (NOT ROUGHPY_ENABLE_IWYU)
target_precompile_headers(RoughPy_Platform PUBLIC
        <boost/multiprecision/gmp.hpp>
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
endif()


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
