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
        src/algebra.cpp
        src/augmented_basis.cpp
        src/augmented_basis.h
        src/basis.cpp
        src/basis_key.cpp
        src/basis_key_type.cpp
        src/basis_key_type.h
        src/context.cpp
        src/free_tensor.cpp
        src/hall_set_size.cpp
        src/hall_set_size.h
        src/hall_word.cpp
        src/hall_word.h
        src/hall_word_iterator_state.cpp
        src/hall_word_iterator_state.h
        src/key_array.cpp
        src/key_iterator_state.cpp
        src/key_range.cpp
        src/key_scalar_array.cpp
        src/key_scalar_stream.cpp
        src/lie.cpp
        src/lie_basis.cpp
        #        src/mutable_vector_element.cpp
        #        src/mutable_vector_element.h
        src/tensor_basis.cpp
        src/tensor_key_iterator_state.cpp
        src/tensor_key_iterator_state.h
        src/tensor_product_basis.cpp
        src/tensor_product_basis.h
        src/tensor_product_basis_key.cpp
        src/tensor_product_basis_key.h
        src/tensor_word.cpp
        src/tensor_word.h
        src/vector.cpp
        src/vector_data.cpp
        src/vector_iterator.cpp
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
        DEPENDENCIES PUBLIC
        Boost::boost
        PRIVATE
        RoughPy_Algebra_Host_Kernels
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
)


add_roughpy_test(Algebra
        SRC
        src/test_vector.cpp
        NEEDS
        RoughPy::Algebra
)
