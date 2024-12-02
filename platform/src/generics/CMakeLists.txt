

set(_header_dir "${ROUGHPY_PLATFORM_INCLUDE_DIR}/generics")

target_sources(RoughPy_Platform PRIVATE
        ${_header_dir}/arithmetic_trait.h
        ${_header_dir}/builtin_trait.h
        ${_header_dir}/comparison_trait.h
        ${_header_dir}/conversion_trait.h
        ${_header_dir}/number_trait.h
        ${_header_dir}/type.h
        ${_header_dir}/values.h

        arithmetic_trait.cpp
        backup_display.cpp
        comparison_trait.cpp
        conversion_trait.cpp
        conversion_factory.cpp
        conversion_impl.h
        multiprecision_types.cpp
        number_trait.cpp
        polynomial_types.cpp
        type.cpp
        type_builtin.cpp
        type_promotion.cpp
        value.cpp
        value_arithmetic.cpp
        value_compare.cpp
        value_math.cpp
)


add_subdirectory(builtin_types)
add_subdirectory(multiprecision_types)
add_subdirectory(polynomial_type)



if (ROUGHPY_BUILD_TESTS)


    add_executable(test_generics
            test_value.cpp
    )

    target_link_libraries(test_generics PRIVATE RoughPy::Platform GTest::gtest)

    setup_roughpy_cpp_tests(test_generics)


endif()