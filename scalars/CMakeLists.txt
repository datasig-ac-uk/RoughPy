cmake_minimum_required(VERSION 3.21)

project(RoughPyScalars
        VERSION 0.0.1
        LANGUAGES CXX
)

include(GNUInstallDirs)
include(GenerateExportHeader)


option(RPY_POLY_SCALARS "Build polynomial scalar types" ON)

set(RPY_USE_MKL ${MKL_FOUND})

configure_file(scalar_blas_defs.h.in scalar_blas_defs.h @ONLY)
add_roughpy_component(Scalars
        SOURCES
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
        src/half_type.h
        src/half_type.cpp
        src/float_type.cpp
        src/float_type.h
        src/double_type.cpp
        src/double_type.h
        src/RationalType.cpp
        src/RationalType.h
        src/standard_random_generator.cpp
        src/standard_random_generator.h
        src/scalar_matrix.cpp
        src/scalar_blas.cpp
        src/float_blas.cpp
        src/float_blas.h
        src/scalar_blas_impl.h
        src/half_random_generator.cpp
        src/half_random_generator.h
        src/b_float_16_type.cpp
        src/b_float_16_type.h
        src/bfloat16_random_generator.cpp
        src/bfloat16_random_generator.h
        src/random_impl.cpp
        src/random_impl.h
        src/rational_poly_scalar_type.cpp
        src/rational_poly_scalar_type.h
        ${CMAKE_CURRENT_BINARY_DIR}/scalar_blas_defs.h
        PUBLIC_HEADERS
        include/roughpy/scalars/scalars_fwd.h
        include/roughpy/scalars/scalar_pointer.h
        include/roughpy/scalars/scalar.h
        include/roughpy/scalars/scalar_type.h
        include/roughpy/scalars/scalar_interface.h
        include/roughpy/scalars/scalar_array.h
        include/roughpy/scalars/owned_scalar_array.h
        include/roughpy/scalars/key_scalar_array.h
        include/roughpy/scalars/scalar_traits.h
        include/roughpy/scalars/random.h
        include/roughpy/scalars/scalar_matrix.h
        include/roughpy/scalars/scalar_blas.h
        include/roughpy/scalars/scalar_type_helper.h
        include/roughpy/scalars.h
        PUBLIC_DEPS
        Boost::boost
        Eigen3::Eigen
        Libalgebra_lite::Libalgebra_lite
        BLAS::BLAS
        LAPACK::LAPACK
        PRIVATE_DEPS
        PCGRandom::pcg_random
        NEEDS
        RoughPy::Core
        RoughPy::Platform
)


if (TARGET Bignum::Bignum)
    set(RPY_RATIONAL_HEADER "gmp.hpp")
    set(RPY_RATIONAL_TYPE "boost::multiprecision::mpq_rational")
    set(RPY_USING_GMP ON)
else ()
    set(RPY_USING_GMP OFF)
    set(RPY_RATIONAL_HEADER "cpp_int.hpp")
    set(RPY_RATIONAL_TYPE "boost::multiprecision::cpp_rational")
endif ()

#configure_file("src/rational_type.h.in" "rational_type.h" NO_SOURCE_PERMISSIONS @ONLY)


add_roughpy_test(scalar
        SRC
        src/ScalarTests.h
        src/test_scalar.cpp
        src/test_scalar_type.cpp
        src/test_float_blas.cpp
        src/test_scalar_array.cpp
        src/test_key_scalar_array.cpp
        src/test_scalar_matrix.cpp
        src/test_pcg_standard_random.cpp
        NEEDS
        RoughPy::Scalars
)
