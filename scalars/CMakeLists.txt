
# Scalars is going to be removed soon, so all this functionality, where appropriate, will
# be moved completely to other parts of the package. For that reason, I'm not going to put
# in a significant amount of effort refactoring this file just the bare minimum.

setup_roughpy_component(Scalars VERSION 0.0.9)


add_library(RoughPy_Scalars SHARED
        src/random/random_impl.cpp
        src/random/random_impl.h
        src/random/standard_random_generator.cpp
        src/random/standard_random_generator.h
        src/scalar/arithmetic.cpp
        src/scalar/arithmetic.h
        src/scalar/casts.cpp
        src/scalar/casts.h
        src/scalar/comparison.cpp
        src/scalar/comparison.h
        src/scalar/do_macro.h
        src/scalar/print.cpp
        src/scalar/print.h
        src/scalar/raw_bytes.cpp
        src/scalar/raw_bytes.h
        src/scalar/serialization.cpp
        src/scalar/serialization.h
        src/scalar/type_promotion.cpp
        src/scalar/type_promotion.h
        src/scalar_helpers/standard_scalar_type.h
        src/types/aprational/ap_rational_type.cpp
        src/types/aprational/ap_rational_type.h
        src/types/apratpoly/ap_rat_poly_type.cpp
        src/types/apratpoly/ap_rat_poly_type.h
        src/types/bfloat16/b_float_16_type.cpp
        src/types/bfloat16/b_float_16_type.h
        src/types/bfloat16/bfloat16_random_generator.cpp
        src/types/bfloat16/bfloat16_random_generator.h
        src/types/double/double_type.cpp
        src/types/double/double_type.h
        src/types/float/float_type.cpp
        src/types/float/float_type.h
        src/types/half/half_random_generator.cpp
        src/types/half/half_random_generator.h
        src/types/half/half_type.cpp
        src/types/half/half_type.h
        src/key_scalar_array.cpp
        src/key_scalar_stream.cpp
        src/random.cpp
        src/scalar.cpp
        src/scalar_array.cpp
        src/scalar_array_view.cpp
        src/scalar_interface.cpp
        src/scalar_serialization.cpp
        src/scalar_stream.cpp
        src/scalar_type.cpp
        src/scalar_type_of.cpp

        include/roughpy/scalars/key_scalar_array.h
        include/roughpy/scalars/key_scalar_stream.h
        include/roughpy/scalars/packed_scalar_type_ptr.h
        include/roughpy/scalars/scalar.h
        include/roughpy/scalars/scalar_array.h
        include/roughpy/scalars/scalar_array_view.h
        include/roughpy/scalars/scalar_interface.h
        include/roughpy/scalars/scalar_serialization.h
        include/roughpy/scalars/scalar_stream.h
        include/roughpy/scalars/scalar_traits.h
        include/roughpy/scalars/scalar_type.h
        include/roughpy/scalars/scalar_types.h
        include/roughpy/scalars/scalars_fwd.h
        include/roughpy/scalars/traits.h
        include/roughpy/scalars.h
)
add_library(RoughPy::Scalars ALIAS RoughPy_Scalars)

target_include_directories(RoughPy_Scalars PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/include/roughpy/scalars
)

target_link_libraries(RoughPy_Scalars PUBLIC
        RoughPy::Platform
        PRIVATE
        RoughPy::PrecompiledHeaders
        PCGRandom::pcg_random
)

set_target_properties(RoughPy_Scalars PROPERTIES ROUGHPY_COMPONENT Scalars)

# TODO: This should be removed, it's not sensible
target_compile_definitions(RoughPy_Scalars PRIVATE "RPY_COMPILING_SCALARS")

set_library_version_properties(RoughPy_Scalars)

generate_export_header(RoughPy_Scalars)

if (ROUGHPY_BUILD_TESTS)

    add_executable(test_scalars
            src/test_key_scalar_array.cpp
            src/test_monomial.cpp
            src/test_pcg_standard_random.cpp
            src/test_scalar.cpp
            src/test_scalar_type.cpp
    )

    target_link_libraries(test_scalars PRIVATE RoughPy::Scalars GTest::gtest)


    setup_roughpy_cpp_tests(test_scalars)

endif ()

