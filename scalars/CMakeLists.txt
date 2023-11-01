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
        include/roughpy/scalars/scalar_type.h
        include/roughpy/scalars/scalar_types.h
        include/roughpy/scalars/scalars_fwd.h
    SOURCES
        src/scalar/arithmetic.cpp
        src/scalar/arithmetic.h
        src/scalar/casts.cpp
        src/key_scalar_array.cpp
        src/key_scalar_stream.cpp
        src/random.cpp
        src/scalar.cpp
        src/scalar_array.cpp
        src/scalar_array_view.cpp
        src/scalar_serialization.cpp
        src/scalar_stream.cpp
        src/scalar_type.cpp
    DEPENDENCIES
        PUBLIC
            Boost::boost
        PRIVATE
            PCGRandom::pcg_random
        NEEDS
            RoughPy::Core
            RoughPy::Platform
            RoughPy::Device
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
