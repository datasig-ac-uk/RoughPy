

setup_roughpy_component(Algebra)

add_library(RoughPy_Algebra SHARED
        src/algebra_base.cpp
        src/algebra_iterator.cpp
        src/basis.cpp
        src/context.cpp
        src/double_lite_context.cpp
        src/float_lite_context.cpp
        src/free_tensor.cpp
        src/free_tensor_interface.cpp
        src/hall_set_size.cpp
        src/hall_set_size.h
        src/lie.cpp
        src/lie_basis.cpp
        src/lie_interface.cpp
        src/lite_context.cpp
        src/lite_context.h
        src/rational_lite_context.cpp
        src/rational_poly_lite_context.cpp
        src/shuffle_tensor.cpp
        src/shuffle_tensor_interface.cpp
        src/tensor_basis.cpp

        include/roughpy/algebra/implementors/algebra_impl.h
        include/roughpy/algebra/implementors/free_tensor_impl.h
        include/roughpy/algebra/implementors/lie_impl.h
        include/roughpy/algebra/implementors/shuffle_tensor_impl.h
        include/roughpy/algebra/interfaces/algebra_interface.h
        include/roughpy/algebra/interfaces/free_tensor_interface.h
        include/roughpy/algebra/interfaces/lie_interface.h
        include/roughpy/algebra/interfaces/shuffle_tensor_interface.h
        include/roughpy/algebra/algebra_base.h
        include/roughpy/algebra/algebra_base_impl.h
        include/roughpy/algebra/algebra_fwd.h
        include/roughpy/algebra/algebra_info.h
        include/roughpy/algebra/algebra_iterator.h
        include/roughpy/algebra/algebra_iterator_impl.h
        include/roughpy/algebra/basis.h
        include/roughpy/algebra/basis_impl.h
        include/roughpy/algebra/basis_info.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/free_tensor.h
        include/roughpy/algebra/interface_traits.h
        include/roughpy/algebra/lie.h
        include/roughpy/algebra/lie_basis.h
        include/roughpy/algebra/linear_operator.h
        include/roughpy/algebra/shuffle_tensor.h
        include/roughpy/algebra/tensor_basis.h
)
add_library(RoughPy::Algebra ALIAS RoughPy_Algebra)


target_include_directories(RoughPy_Platform PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/algebra
)

target_link_libraries(RoughPy_Algebra
        PUBLIC
        RoughPy::Platform
        RoughPy::Scalars
        PRIVATE
        Libalgebra_lite::Libalgebra_lite
)

set_target_properties(RoughPy_Algebra PROPERTIES ROUGHPY_COMPONENT Algebra)

set_library_version_properties(RoughPy_Algebra)


add_subdirectory(src/libalgebra_lite_internal)


if (ROUGHPY_BUILD_TESTS)

        add_executable(test_algebra
                src/ContextFixture.cpp
                src/ContextFixture.h
                src/test_lie.cpp
        )

        target_link_libraries(test_algebra PRIVATE RoughPy::Algebra GTest::gtest)

        setup_roughpy_tests(test_algebra)

endif()


