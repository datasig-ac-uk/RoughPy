cmake_minimum_required(VERSION 3.21)


project(RoughPy_Algebra
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)


add_library(RoughPy_Algebra_headers INTERFACE)
target_include_directories(RoughPy_Algebra_headers INTERFACE include)
target_link_libraries(RoughPy_Algebra_headers INTERFACE
        RoughPy::Core
        RoughPy::Platform
        Boost::boost
)

set(ROUGHPY_ALGEBRA_PRIVATE_HEADERS ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/algebra)
set(ROUGHPY_SCALARS_PRIVATE_HEADERS ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/scalars)

add_subdirectory(src/scalars)
add_subdirectory(src/algebra/basis_keys)


add_roughpy_component(Algebra
        SOURCES
        src/algebra/algebra.cpp
        src/algebra/basis.cpp
        src/algebra/context.cpp
        src/algebra/dense_vector_factory.cpp
        src/algebra/dense_vector_factory.h
        src/algebra/free_tensor.cpp
        src/algebra/free_tensor_multiplication.cpp
        src/algebra/key_array.cpp
        #        src/algebra/key_range.cpp
        #        src/algebra/key_scalar_array.cpp
        #        src/algebra/key_scalar_stream.cpp
        src/algebra/lie.cpp
        #        src/mutable_vector_element.cpp
        #        src/mutable_vector_element.h
        src/algebra/vector.cpp
        src/algebra/vector_factory.cpp
        src/algebra/vector_data.cpp
        src/algebra/vector_iterator.cpp
        src/scalars/builtin_types.cpp
        PUBLIC_HEADERS
        include/roughpy/algebra/kernels/dense_free_tensor_multiplication.h
        include/roughpy/algebra/algebra.h
        include/roughpy/algebra/algebra_fwd.h
        include/roughpy/algebra/basis.h
        include/roughpy/algebra/basis_keys.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/free_tensor.h
        include/roughpy/algebra/key_algorithms.h
        include/roughpy/algebra/key_array.h
        #        include/roughpy/algebra/key_scalar_array.h
        #        include/roughpy/algebra/key_scalar_stream.h
        include/roughpy/algebra/lie.h
        include/roughpy/algebra/shuffle_tensor.h
        include/roughpy/algebra/vector.h
        include/roughpy/algebra/vector_iterator.h
        src/scalars/scalar_types/polynomial/monomial.h
        include/roughpy/scalars/scalar_array.h
        include/roughpy/scalars/scalar_stream.h
        include/roughpy/scalars/scalar_value.h
        include/roughpy/scalars/scalar_vector.h
        include/roughpy/scalars/scalars_fwd.h
        DEPENDENCIES PUBLIC
        Boost::boost
        RoughPy_Algebra_headers
        PRIVATE
        RoughPy_scalars
        RoughPy_Scalar_Type_half
        RoughPy_Scalar_Type_bfloat16
        RoughPy_Scalar_Type_rational
        RoughPy_Scalar_Type_polynomial
        RoughPy_tensor_word
        RoughPy_lie_word
        RoughPy_index_key
        NEEDS
        RoughPy::Core
        RoughPy::Platform
)

add_subdirectory(src/algebra)

add_roughpy_test(Algebra
        SRC
        src/algebra/test_vector.cpp
        NEEDS
        RoughPy::Algebra
)
