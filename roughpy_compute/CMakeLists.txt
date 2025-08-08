
cmake_minimum_required(VERSION 3.30)

project(RoughPy-Compute LANGUAGES CXX)



option(ROUGHPY_BUILD_TESTS "build tests for roughpy_compute" ON)


add_library(roughpy_compute INTERFACE
        common/architecture.hpp
        common/basis.hpp
        common/cache_array.hpp
        common/operations.hpp
        dense/views.hpp
        dense/basic/free_tensor_antipode.hpp
        dense/basic/free_tensor_fma.hpp
        dense/basic/free_tensor_inplace_mul.hpp
        dense/basic/shuffle_tensor_product.hpp
        dense/basic/vector_addition.hpp
        dense/basic/vector_inplace_addition.hpp
        dense/basic/vector_inplace_scalar_multiply.hpp
        dense/basic/vector_scalar_multiply.hpp
        dense/intermediate/free_tensor_exp.hpp
        dense/intermediate/free_tensor_fmexp.hpp
        dense/intermediate/free_tensor_log.hpp
)





target_include_directories(roughpy_compute INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
)

if (ROUGHPY_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
    enable_testing()
    include(GoogleTest)
    include(FetchContent)

    FetchContent_Declare(
            LibalgebraLite
            GIT_REPOSITORY https://github.com/datasig-ac-uk/libalgebra-lite.git
            GIT_TAG 5adf56d2211df9faec7b0990ce332cfae0c42396
    )

    set(LIBALGEBRA_LITE_RATIONAL_COEFFS ON CACHE INTERNAL "we need rational coefficients")
    FetchContent_MakeAvailable(LibalgebraLite)

    add_executable(roughpy_compute_tests
            testing/tensor_helpers.hpp
            dense/basic/vector_addition_tests.cpp
            dense/basic/vector_inplace_addition_tests.cpp
            dense/basic/free_tensor_fma_tests.cpp
    )


    target_link_libraries(roughpy_compute_tests PRIVATE
            GTest::gtest_main
            roughpy_compute
            Libalgebra_lite::Libalgebra_lite)

    gtest_discover_tests(roughpy_compute_tests)
endif()