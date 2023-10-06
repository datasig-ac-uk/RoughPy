// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_

#include "algebra_base.h"
#include "algebra_bundle.h"
#include "tensor_basis.h"

namespace rpy {
namespace algebra {

// extern template class AlgebraInterface<ShuffleTensor, TensorBasis>;

class RPY_EXPORT ShuffleTensorInterface
    : public AlgebraInterface<ShuffleTensor, TensorBasis>
{
    using base_t = AlgebraInterface<ShuffleTensor, TensorBasis>;

public:
    using base_t::base_t;
};

namespace traits {
namespace dtl {

template <>
struct algebra_of_impl<ShuffleTensorInterface> {
    using type = ShuffleTensor;
};

template <>
struct basis_of_impl<ShuffleTensorInterface> {
    using type = TensorBasis;
};

}// namespace dtl
}// namespace traits
// extern template class AlgebraBase<ShuffleTensorInterface>;

class RPY_EXPORT ShuffleTensor : public AlgebraBase<ShuffleTensorInterface>
{
    using base_t = AlgebraBase<ShuffleTensorInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensor;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_EXTERN_SERIALIZE_CLS(ShuffleTensor)

class ShuffleTensorBundle;

// extern template class BundleInterface<ShuffleTensorBundle, ShuffleTensor,
//                                       ShuffleTensor>;

class RPY_EXPORT ShuffleTensorBundleInterface
    : public BundleInterface<ShuffleTensorBundle, ShuffleTensor, ShuffleTensor>
{
};

namespace traits { namespace dtl {

template <>
struct algebra_of_impl<ShuffleTensorBundleInterface> {
    using type = ShuffleTensorBundle;
};

template <>
struct basis_of_impl<ShuffleTensorBundleInterface> {
    using type = TensorBasis;
};

}}


// extern template class AlgebraBundleBase<ShuffleTensorBundleInterface>;

class RPY_EXPORT ShuffleTensorBundle
    : public AlgebraBundleBase<ShuffleTensorBundleInterface>
{
    using base_t = AlgebraBundleBase<ShuffleTensorBundleInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensorBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_EXTERN_SERIALIZE_CLS(ShuffleTensorBundle)

}// namespace algebra
}// namespace rpy
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensor,
        rpy::serial::specialization::member_serialize
)
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensorBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_
