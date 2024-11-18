



add_library(RoughPy_Generics_BuiltinType STATIC
        builtin_type.h
        builtin_type_ids.h
        builtin_type_methods.h
        conversion_factory.cpp
        conversion_factory.h
        double_type.cpp
        double_type.h
        float_type.cpp
        float_type.h
        signed_int_type.cpp
        signed_int_type.h
        unsigned_int_type.cpp
        unsigned_int_type.h
)

set_target_properties(RoughPy_Generics_BuiltinType PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(RoughPy_Generics_BuiltinType PRIVATE
        ${ROUGHPY_PLATFORM_INCLUDE_DIR}/../
        ${CMAKE_CURRENT_BINARY_DIR}/../../..
)

target_link_libraries(RoughPy_Generics_BuiltinType PRIVATE
        RoughPy::Core
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
