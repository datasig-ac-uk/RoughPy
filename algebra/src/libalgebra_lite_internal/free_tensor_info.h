//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_FREE_TENSOR_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_FREE_TENSOR_INFO_H


#include "algebra_info.h"
#include "tensor_basis_info.h"

#include "vector_type_helper.h"
#include <roughpy/core/traits.h>

#include <libalgebra_lite/free_tensor.h>

namespace rpy { namespace algebra {

template <typename Coeffs, template <typename, typename> class VType,
          template <typename> class Storage>
struct algebra_info<FreeTensor, lal::free_tensor<Coeffs, VType, Storage>> {

    /// The actual type of the algebra implementation
    using algebra_type = lal::free_tensor<Coeffs, VType, Storage>;

    /// The wrapping roughpy algebra type
    using wrapper_type = FreeTensor;

    /// The basis type of the implementation
    using basis_type = typename lal::tensor_basis;

    /// The roughpy key type used in the wrapper
    using key_type = typename TensorBasis::key_type;

    /// Basis traits for querying the basis
    using basis_traits = BasisInfo<TensorBasis, basis_type>;

    /// Scalar type in the implementation
    using scalar_type = typename Coeffs::scalar_type;

    /// Rational type, default to scalar type
    using rational_type = typename Coeffs::rational_type;

    /// Reference type - currently unused
    using reference = scalar_type &;

    /// Const reference type - currently unused
    using const_reference = const scalar_type &;

    /// Pointer type - currently unused
    using pointer = scalar_type *;

    /// Const pointer type - currently unused
    using const_pointer = const scalar_type *;

    /// Get the rpy ScalarType for the scalars in this algebra
    static const scalars::ScalarType *ctype() noexcept { return scalars::ScalarType::of<scalar_type>(); }

    /// Get the storage type for this algebra.
    static constexpr VectorType vtype() noexcept { return dtl::vector_type_helper<VType>::vtype; }

    /// Get the basis for this algebra
    static const basis_type &basis(const algebra_type &instance) noexcept { return instance.basis(); }

    /// Get the maximum degree of non-zero elements in this algebra
    static deg_t degree(const algebra_type &instance) noexcept { return instance.degree(); }

    /// Create a new algebra instance with the same make-up as this argument
    static algebra_type create_like(const algebra_type &instance) {
        return algebra_type(instance.get_basis(), instance.multiplication());
    }
};
}}



#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_FREE_TENSOR_INFO_H
