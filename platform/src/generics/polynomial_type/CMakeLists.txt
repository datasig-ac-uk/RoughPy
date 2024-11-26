



target_sources(RoughPy_Platform PRIVATE
    indeterminate.h
    indeterminate.cpp
    monomial.cpp
    monomial.h
        polynomial.cpp
        polynomial.h
)


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


endif()