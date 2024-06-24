cmake_minimum_required(VERSION 3.21)


project(RoughPy_Algebra
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)


add_library(RoughPy_Algebra_Host_Kernels STATIC
        src/host_kernels/common.h
        src/host_kernels/vector_binary_operator.h
        src/host_kernels/vector_unary_operator.h
        src/host_kernels/algebra_binary_kernel.h
)

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

set_target_properties(RoughPy_Algebra_Host_Kernels PROPERTIES
        LINKER_LANGUAGE CXX
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(RoughPy_Algebra_Host_Kernels PRIVATE RoughPy::Core RoughPy::Platform RoughPy_Device_Support)


add_roughpy_component(Algebra
        SOURCES
        src/kernels/addition_kernel.cpp
        src/kernels/addition_kernel.h
        src/kernels/arg_data.h
        src/kernels/argument_specs.h
        src/kernels/common.h
        src/kernels/fused_add_left_scalar_multiply.cpp
        src/kernels/fused_add_left_scalar_multiply.h
        src/kernels/fused_add_right_scalar_multiply.cpp
        src/kernels/fused_add_right_scalar_multiply.h
        src/kernels/fused_sub_right_scalar_multiply_kernel.cpp
        src/kernels/fused_sub_right_scalar_multiply_kernel.h
        src/kernels/generic_inplace_multiplication.cpp
        src/kernels/generic_inplace_multiplication.h
        src/kernels/generic_kernel.h
        src/kernels/generic_kernel_MV_CS.cpp
        src/kernels/generic_kernel_MV_CS.h
        src/kernels/generic_kernel_CV_MS.cpp
        src/kernels/generic_kernel_CV_MS.h
        src/kernels/generic_kernel_MV_CV.cpp
        src/kernels/generic_kernel_MV_CV.h
        src/kernels/generic_kernel_MV_CV_CS.cpp
        src/kernels/generic_kernel_MV_CV_CS.h
        src/kernels/generic_kernel_MV_CV_CV.cpp
        src/kernels/generic_kernel_MV_CV_CV.h
        src/kernels/generic_multiplication_kernel.cpp
        src/kernels/generic_multiplication_kernel.h
        src/kernels/get_kernel.cpp
        src/kernels/inplace_addition_kernel.cpp
        src/kernels/inplace_addition_kernel.h
        src/kernels/inplace_left_scalar_multiply_kernel.cpp
        src/kernels/inplace_left_scalar_multiply_kernel.h
        src/kernels/inplace_right_scalar_multiply_kernel.cpp
        src/kernels/inplace_right_scalar_multiply_kernel.h
        src/kernels/inplace_subtraction_kernel.cpp
        src/kernels/inplace_subtraction_kernel.h
        src/kernels/kernel.h
        src/kernels/left_scalar_multiply_kernel.cpp
        src/kernels/left_scalar_multiply_kernel.h
        src/kernels/right_scalar_multiply_kernel.cpp
        src/kernels/right_scalar_multiply_kernel.h
        src/kernels/sparse_helpers.h
        src/kernels/sparse_write_kernel.cpp
        src/kernels/sparse_write_kernel
        src/kernels/subtraction_kernel.cpp
        src/kernels/subtraction_kernel.h
        src/kernels/uminus_kernel.cpp
        src/kernels/uminus_kernel.h
    src/algebra/algebra.cpp
    src/algebra/augmented_basis.cpp
    src/algebra/augmented_basis.h
    src/algebra/basis.cpp
    src/algebra/basis_key.cpp
    src/algebra/basis_key_type.cpp
    src/algebra/basis_key_type.h
    src/algebra/context.cpp
    src/algebra/free_tensor.cpp
    src/algebra/hall_set_size.cpp
    src/algebra/hall_set_size.h
    src/algebra/hall_word.cpp
    src/algebra/hall_word.h
    src/algebra/hall_word_iterator_state.cpp
    src/algebra/hall_word_iterator_state.h
    src/algebra/key_array.cpp
    src/algebra/key_iterator_state.cpp
    src/algebra/key_range.cpp
    src/algebra/key_scalar_array.cpp
    src/algebra/key_scalar_stream.cpp
    src/algebra/lie.cpp
    src/algebra/lie_basis.cpp
        #        src/mutable_vector_element.cpp
        #        src/mutable_vector_element.h
    src/algebra/tensor_basis.cpp
    src/algebra/tensor_key_iterator_state.cpp
    src/algebra/tensor_key_iterator_state.h
    src/algebra/tensor_product_basis.cpp
    src/algebra/tensor_product_basis.h
    src/algebra/tensor_product_basis_key.cpp
    src/algebra/tensor_product_basis_key.h
    src/algebra/tensor_word.cpp
    src/algebra/tensor_word.h
    src/algebra/vector.cpp
    src/algebra/vector_data.cpp
    src/algebra/vector_iterator.cpp
        PUBLIC_HEADERS
        include/roughpy/algebra/algebra.h
        include/roughpy/algebra/algebra_fwd.h
        include/roughpy/algebra/basis.h
        include/roughpy/algebra/basis_key.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/free_tensor.h
        include/roughpy/algebra/graded_vector.h
        include/roughpy/algebra/key_algorithms.h
        include/roughpy/algebra/key_array.h
        include/roughpy/algebra/key_scalar_array.h
        include/roughpy/algebra/key_scalar_stream.h
        include/roughpy/algebra/lie.h
        include/roughpy/algebra/lie_basis.h
        include/roughpy/algebra/shuffle_tensor.h
        include/roughpy/algebra/tensor_basis.h
        include/roughpy/algebra/unital_algebra.h
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
        RoughPy_Algebra_Host_Kernels
        NEEDS
        RoughPy::Core
        RoughPy::Platform
)


add_roughpy_test(Algebra
        SRC
    src/algebra/test_vector.cpp
        NEEDS
        RoughPy::Algebra
)
