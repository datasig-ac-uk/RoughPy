// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SHUFFLE_TENSOR_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SHUFFLE_TENSOR_INFO_H

#include "vector_type_helper.h"
#include <roughpy/algebra/algebra_info.h>

#include <roughpy/core/traits.h>

#include <libalgebra_lite/shuffle_tensor.h>

namespace rpy {
namespace algebra {

template <
        typename Coeffs, template <typename, typename> class VType,
        template <typename> class Storage>
struct algebra_info<
        ShuffleTensor, lal::shuffle_tensor<Coeffs, VType, Storage>> {
    /// The actual type of the algebra implementation
    using algebra_type = lal::shuffle_tensor<Coeffs, VType, Storage>;

    /// The wrapping roughpy algebra type
    using wrapper_type = ShuffleTensor;

    /// The basis type of the implementation
    using basis_type = lal::tensor_basis;

    /// The roughpy key type used in the wrapper
    using key_type = typename TensorBasis::key_type;

    /// Basis traits for querying the basis
    using basis_traits = BasisInfo<TensorBasis, basis_type>;

    /// Scalar type in the implementation
    using scalar_type = typename Coeffs::scalar_type;

    /// Rational type, default to scalar type
    using rational_type = typename Coeffs::rational_type;

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
        return dtl::vector_type_helper<VType>::vtype;
    }

    /// Get the basis for this algebra
    static const basis_type& basis(const algebra_type& instance) noexcept
    {
        return instance.basis();
    }

    /// Get the maximum degree of non-zero elements in this algebra
    static deg_t degree(const algebra_type& instance) noexcept
    {
        return instance.degree();
    }

    /// Create a new algebra instance with the same make-up as this argument
    static algebra_type create_like(const algebra_type& instance)
    {
        return algebra_type(instance.get_basis(), instance.multiplication());
    }
};

}// namespace algebra
}// namespace rpy
#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SHUFFLE_TENSOR_INFO_H
