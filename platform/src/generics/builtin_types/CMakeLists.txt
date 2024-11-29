



target_sources(RoughPy_Platform PRIVATE
        builtin_type.h
        builtin_type_ids.h
        builtin_type_methods.h
        conversion_factory.h
        conversion_helpers.h
        double_type.cpp
        double_type.h
        float_type.cpp
        float_type.h
        signed_int_type.cpp
        signed_int_type.h
        unsigned_int_type.cpp
        unsigned_int_type.h
)


if (ROUGHPY_BUILD_TESTS)

    add_executable(test_builtin_types
            test_builtin_conversions.cpp
            test_double_type.cpp
            test_float_type.cpp
    )

    target_link_libraries(test_builtin_types PRIVATE
            RoughPy_Platform GTest::gtest
    )

    setup_roughpy_cpp_tests(test_builtin_types)


endif()
