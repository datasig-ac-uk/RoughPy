#ifndef ROUGHPY_LA_CONTEXT_SHUFFLE_TENSOR_INFO_H_
#define ROUGHPY_LA_CONTEXT_SHUFFLE_TENSOR_INFO_H_

#include <roughpy/algebra/algebra_info.h>
#include <roughpy/algebra/shuffle_tensor.h>

#include <libalgebra/tensor.h>

#include "vector_type_helper.h"

namespace rpy {
namespace algebra {

template <typename Coeffs, alg::DEG Width, alg::DEG Depth,
          /*template <typename, typename, typename...> class VType,*/
          typename... Args>
struct algebra_info<
        ShuffleTensor,
        alg::shuffle_tensor<Coeffs, Width, Depth, /*VType,*/ Args...>> {
    /// The actual type of the algebra implementation
    using algebra_type
            = alg::shuffle_tensor<Coeffs, Width, Depth, /*VType,*/ Args...>;

    /// The wrapping roughpy algebra type
    using wrapper_type = ShuffleTensor;

    /// The basis type of the implementation
    using basis_type = alg::tensor_basis<Width, Depth>;

    /// The roughpy key type used in the wrapper
    using key_type = typename TensorBasis::key_type;

    /// Basis traits for querying the basis
    using basis_traits = BasisInfo<TensorBasis, basis_type>;

    /// Scalar type in the implementation
    using scalar_type = typename Coeffs::S;

    /// Rational type, default to scalar type
    using rational_type = typename Coeffs::Q;

    /// Reference type - currently unused
    using reference = scalar_type&;

    /// Const reference type - currently unused
    using const_reference = const scalar_type&;

    /// Pointer type - currently unused
    using pointer = scalar_type*;

    /// Const pointer type - currently unused
    using const_pointer = const scalar_type*;

    /// Get the rpy ScalarType for the scalars in this algebra
    static const scalars::ScalarType* ctype() noexcept
    {
        return scalars::ScalarType::of<scalar_type>();
    }

    /// Get the storage type for this algebra.
    static constexpr VectorType vtype() noexcept
    {
        return VectorType::Dense;
        //        return dtl::la_vector_type_helper<VType>::vtype;
    }

    /// Get the basis for this algebra
    static const basis_type& basis(const algebra_type& instance) noexcept
    {
        return instance.basis;
    }

    /// Get the maximum degree of non-zero elements in this algebra
    static deg_t degree(const algebra_type& instance) noexcept
    {
        return instance.degree();
    }

    /// Create a new algebra instance with the same make-up as this argument
    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type();
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_LA_CONTEXT_SHUFFLE_TENSOR_INFO_H_
