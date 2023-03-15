cmake_minimum_required(VERSION 3.18)

project(RoughPyScalars
        VERSION 0.0.1
        LANGUAGES CXX
        )

include(GNUInstallDirs)
include(GenerateExportHeader)

find_package(Boost 1.64 CONFIG REQUIRED COMPONENTS headers)

option(RPY_SCALARS_SHARED "Build shared RoughPy scalars module" ON)


if (RPY_SCALARS_SHARED)
    add_library(RoughPy_Scalars SHARED)
else()
    add_library(RoughPy_Scalars STATIC)
endif()

add_library(RoughPy::Scalars ALIAS RoughPy_Scalars)

target_link_libraries(RoughPy_Scalars PUBLIC RoughPy::Config Boost::boost)

set(ROUGHPY_SCALARS_PUBLIC_HEADERS
        include/roughpy/scalars/scalars_fwd.h
        include/roughpy/scalars/scalar_pointer.h
        include/roughpy/scalars/scalar.h
        include/roughpy/scalars/scalar_type.h
        include/roughpy/scalars/scalar_interface.h
        include/roughpy/scalars/scalar_array.h
        include/roughpy/scalars/owned_scalar_array.h
        include/roughpy/scalars/key_scalar_array.h
        include/roughpy/scalars/scalar_traits.h
        include/roughpy/scalars.h
        )

target_sources(RoughPy_Scalars
        PUBLIC
            ${ROUGHPY_SCALARS_PUBLIC_HEADERS}
        PRIVATE
            src/scalar_type.cpp
            src/scalar_pointer.cpp
            src/scalar_interface.cpp
            src/scalar.cpp
            src/scalar_pointer.cpp
            src/scalar.cpp
            src/scalar_interface.cpp
            src/scalar_array.cpp
            src/owned_scalar_array.cpp
            src/key_scalar_array.cpp
            src/scalar_stream.cpp
            src/standard_scalar_type.h
            src/float_type.cpp
            src/float_type.h
            src/double_type.cpp
            src/double_type.h
            src/RationalType.cpp
            src/RationalType.h
        )



target_include_directories(RoughPy_Scalars
        PRIVATE
            "include/roughpy/scalars"
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

generate_export_header(RoughPy_Scalars)


set_target_properties(RoughPy_Scalars PROPERTIES
        PUBLIC_HEADER "${ROUGHPY_SCALARS_PUBLIC_HEADERS}"

        )

if (TARGET Bignum::Bignum)
    set(RPY_RATIONAL_HEADER "gmp.hpp")
    set(RPY_RATIONAL_TYPE "boost::multiprecision::mpq_rational")
    target_link_libraries(RoughPy_Scalars PUBLIC Bignum::Bignum)
else()
    set(RPY_RATIONAL_HEADER "cpp_int.hpp")
    set(RPY_RATIONAL_TYPE "boost::multiprecision::cpp_rational")
endif()

configure_file("src/rational_type.h.in" "rational_type.h" NO_SOURCE_PERMISSIONS @ONLY)


add_roughpy_test(scalar
        LINK_COMPONENT
        SRC
            src/ScalarTests.h
            src/test_scalar.cpp
            src/test_scalar_type.cpp
        )
