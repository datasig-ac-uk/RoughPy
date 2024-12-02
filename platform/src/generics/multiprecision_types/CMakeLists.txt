



target_sources(RoughPy_Platform PRIVATE
        conversion_helpers.h
        rational_type.cpp
        rational_type.h
        rational_arithmetic.cpp
        rational_arithmetic.h
        rational_comparison.cpp
        rational_comparison.h
        rational_conversion.cpp
        rational_conversion.h
        rational_number.cpp
        rational_number.h
        mpz_hash.h
        mpq_string_rep.h
        integer_type.cpp
        integer_type.h
        integer_arithmetic.cpp
        integer_arithmetic.h
        integer_comparison.cpp
        integer_comparison.h
        integer_conversion.cpp
        integer_conversion.h
        integer_number.cpp
        integer_number.h
        float_type.cpp
        float_type.h
        float_arithmetic.cpp
        float_arithmetic.h
        float_comparison.cpp
        float_comparison.h
        float_conversion.cpp
        float_conversion.h
        float_number.cpp
        float_number.h
        multiprecision_type_ids.h
)







target_link_libraries(RoughPy_Platform PRIVATE
        GMP::GMP
        MPFR::MPFR
)


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_multiprecision_types
            test_rational_type.cpp
            test_integer_type.cpp
            test_float_type.cpp
    )

    target_link_Libraries(test_multiprecision_types PRIVATE
            RoughPy_Platform
            GTest::gtest
    )

    setup_roughpy_cpp_tests(test_multiprecision_types)

endif()

