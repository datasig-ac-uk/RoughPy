cmake_minimum_required(VERSION 3.21)

project(RoughPyScalars
        VERSION 0.0.9
        LANGUAGES CXX)



add_roughpy_component(Scalars
    PUBLIC_HEADERS
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
    SOURCES
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
    DEPENDENCIES
        PUBLIC
            Boost::boost
        PRIVATE
            PCGRandom::pcg_random
        NEEDS
            RoughPy::Core
            RoughPy::Platform
)

target_precompile_headers(RoughPy_Scalars PRIVATE
        include/roughpy/scalars/scalars_fwd.h
        include/roughpy/scalars/scalar_types.h
        <roughpy/platform/devices.h>
)

add_roughpy_test(scalars
    SRC
        src/test_key_scalar_array.cpp
        src/test_monomial.cpp
        src/test_pcg_standard_random.cpp
        src/test_scalar.cpp
        src/test_scalar_type.cpp
    NEEDS
        RoughPy::Scalars
)
