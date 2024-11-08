



add_library(Libalgebra_lite SHARED
        algebra/lie_multiplier.cpp
        algebra/polynomial_multiplier.cpp
        algebra/polynomial.cpp
        algebra/half_shuffle_multiplier.cpp
        algebra/free_tensor_multiplier.cpp
        algebra/shuffle_multiplier.cpp
        algebra/polynomial_ring.cpp
        basis/hall_set.cpp
        basis/tensor_basis.cpp
        basis/monomial.cpp
        basis/polynomial_basis.cpp
        basis/unpacked_tensor_word.cpp
        coefficients/floating_fields.cpp
        detail/integer_maths.h
        detail/macros.h
        detail/notnull.h
        detail/traits.h
        algebra.h
        basis.h
        basis_traits.h
        coefficients.h
        dense_vector.h
        free_tensor.h
        hall_set.h
        implementation_types.h
        index_key.h
        key_range.h
        lie.h
        maps.cpp
        maps.h
        operators.h
        packed_integer.h
        polynomial.h
        polynomial_basis.h
        registry.h
        shuffle_tensor.h
        sparse_vector.h
        tensor_basis.h
        unpacked_tensor_word.h
        vector.h
        vector_base.h
        vector_bundle.h
        vector_traits.h
)
add_library(Libalgebra_lite::Libalgebra_lite ALIAS Libalgebra_lite)

generate_export_header(Libalgebra_lite)


target_include_directories(Libalgebra_lite
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(Libalgebra_lite PRIVATE
        Boost::headers
)

set_property(TARGET Libalgebra_lite PROPERTY C_INCLUDE_WHAT_YOU_USE)
set_property(TARGET Libalgebra_lite PROPERTY CXX_INCLUDE_WHAT_YOU_USE)

