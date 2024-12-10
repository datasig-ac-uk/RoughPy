



target_sources(RoughPy_Platform PRIVATE
        conversion_helpers.h
        indeterminate.h
        indeterminate.cpp
        monomial.cpp
        monomial.h
        polynomial.cpp
        polynomial.h
        polynomial_arithmetic.cpp
        polynomial_arithmetic.h
        polynomial_comparison.cpp
        polynomial_comparison.h
        polynomial_conversion.cpp
        polynomial_conversion.h
        polynomial_number.cpp
        polynomial_number.h
        polynomial_type.cpp
        polynomial_type.h
)

target_link_libraries(RoughPy_Platform PRIVATE
    ctre::ctre)

if (ROUGHPY_BUILD_TESTS)

    add_executable(test_polynomial
            test_indeterminate.cpp
            test_monomial.cpp
            test_polynomial.cpp
            indeterminate.cpp
            indeterminate.h
            monomial.cpp
            monomial.h
            polynomial.cpp
            polynomial.h
    )

    target_include_directories(test_polynomial PRIVATE
            ${ROUGHPY_PLATFORM_SOURCE}
    )

    target_link_libraries(test_polynomial PRIVATE
            RoughPy::Core
            Boost::headers
            GMP::GMP
            GTest::gtest
    )

    setup_roughpy_cpp_tests(test_polynomial)

    add_executable(test_polynomial_type
            test_polynomial_type.cpp
    )

    target_link_libraries(test_polynomial_type PRIVATE RoughPy::Platform GTest::gtest)

    setup_roughpy_cpp_tests(test_polynomial_type)

endif ()