cmake_minimum_required(VERSION 3.19)


project(RoughPy_Algebra
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)

find_package(Boost 1.64 CONFIG REQUIRED COMPONENTS headers)

option(RPY_ALGEBRA_SHARED "Build shared RoughPy algebra module" ON)


if (NOT TARGET Libalgebra_lite::Libalgebra_lite)
    find_package(Libalgebra_Lite CONFIG REQUIRED)
endif()


if (RPY_ALGEBRA_SHARED)
    add_library(RoughPy_Algebra SHARED)
else()
    add_library(RoughPy_Algebra STATIC)
endif()

add_library(RoughPy::Algebra ALIAS RoughPy_Algebra)

target_link_libraries(RoughPy_Algebra PUBLIC
        RoughPy::Config
        RoughPy::Scalars
        Libalgebra_lite::Libalgebra_lite
        Boost::boost
        )

set(ROUGHPY_ALGEBRA_PUBLIC_HEADERS
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
        include/roughpy/algebra/basis_impl.h
        include/roughpy/algebra/free_tensor.h
        include/roughpy/algebra/free_tensor_impl.h
        include/roughpy/algebra/lie.h
        include/roughpy/algebra/shuffle_tensor.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/fallback_operations.h
)

target_sources(RoughPy_Algebra
        PUBLIC
            ${ROUGHPY_ALGEBRA_PUBLIC_HEADERS}
        PRIVATE
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
            src/libalgebra_lite_internal/dense_vector_iterator.h
            src/libalgebra_lite_internal/free_tensor_info.h
            src/libalgebra_lite_internal/lie_basis_info.h
            src/libalgebra_lite_internal/lie_info.h
            src/libalgebra_lite_internal/lite_vector_selector.h
            src/libalgebra_lite_internal/shuffle_tensor_info.h
            src/libalgebra_lite_internal/sparse_mutable_ref_scalar_trait.h
            src/libalgebra_lite_internal/sparse_vector_iterator.h
            src/libalgebra_lite_internal/tensor_basis_info.h
            src/libalgebra_lite_internal/vector_type_helper.h
        )



target_include_directories(RoughPy_Algebra
        PRIVATE
            "include/roughpy/algebra"
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )

generate_export_header(RoughPy_Algebra)


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
        LINK_COMPONENT
        )
