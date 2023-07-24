cmake_minimum_required(VERSION 3.21)


project(RoughPy_Algebra
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)


if (NOT TARGET Libalgebra_lite::Libalgebra_lite)
    find_package(Libalgebra_Lite CONFIG REQUIRED)
endif ()

add_roughpy_component(Algebra
        SOURCES
        src/basis.cpp
        src/algebra_base.cpp
        src/context.cpp
        src/free_tensor.cpp
        src/lie.cpp
        src/shuffle_tensor.cpp
        src/hall_set_size.cpp
        src/hall_set_size.h
        src/algebra_iterator.cpp
        src/lite_context.cpp
        src/lite_context.h
        src/float_lite_context.cpp
        src/double_lite_context.cpp
        src/rational_lite_context.cpp
        src/rational_poly_lite_context.cpp
        src/libalgebra_lite_internal/algebra_type_caster.h
        src/libalgebra_lite_internal/algebra_type_helper.h
        src/libalgebra_lite_internal/dense_vector_iterator.h
        src/libalgebra_lite_internal/free_tensor_info.h
        src/libalgebra_lite_internal/lie_basis_info.h
        src/libalgebra_lite_internal/lie_info.h
        src/libalgebra_lite_internal/lite_vector_selector.h
        src/libalgebra_lite_internal/shuffle_tensor_info.h
        src/libalgebra_lite_internal/sparse_mutable_ref_scalar_trait.h
        src/libalgebra_lite_internal/sparse_vector_iterator.h
        src/libalgebra_lite_internal/tensor_basis_info.h
        src/libalgebra_lite_internal/unspecified_algebra_binary_op.h
        src/libalgebra_lite_internal/vector_storage_type.h
        src/libalgebra_lite_internal/vector_type_helper.h
        src/tensor_basis.cpp
        src/lie_basis.cpp
        PUBLIC_HEADERS
        include/roughpy/algebra/algebra_fwd.h
        include/roughpy/algebra/algebra_base.h
        include/roughpy/algebra/algebra_impl.h
        include/roughpy/algebra/algebra_info.h
        include/roughpy/algebra/algebra_iterator.h
        include/roughpy/algebra/algebra_iterator_impl.h
        include/roughpy/algebra/algebra_bundle.h
        include/roughpy/algebra/algebra_bundle_impl.h
        include/roughpy/algebra/linear_operator.h
        include/roughpy/algebra/basis.h
        include/roughpy/algebra/basis_info.h
        include/roughpy/algebra/basis_impl.h
        include/roughpy/algebra/bundle_info.h
        include/roughpy/algebra/lie_basis.h
        include/roughpy/algebra/tensor_basis.h
        include/roughpy/algebra/free_tensor.h
        include/roughpy/algebra/free_tensor_fwd.h
        include/roughpy/algebra/free_tensor_impl.h
        include/roughpy/algebra/lie_fwd.h
        include/roughpy/algebra/lie.h
        include/roughpy/algebra/shuffle_tensor_fwd.h
        include/roughpy/algebra/shuffle_tensor.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/fallback_operations.h
        include/roughpy/algebra/algebra_base_impl.h
        include/roughpy/algebra/algebra_bundle_base_impl.h
        PUBLIC_DEPS
        Boost::boost
        Libalgebra_lite::Libalgebra_lite
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        )


add_roughpy_test_helper(AlgebraMock STATIC
        SRCS
        src/testing/mock_context.cpp
        src/testing/mock_context.h
        DEPS
        RoughPy::Algebra
        )

add_roughpy_test(algebra
        SRC
        src/ContextFixture.cpp
        src/ContextFixture.h
        src/test_lie.cpp
        NEEDS
        RoughPy::Algebra
        )
