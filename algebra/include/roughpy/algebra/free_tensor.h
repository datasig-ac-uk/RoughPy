// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H_

#include "algebra_base.h"
#include "algebra_bundle.h"


#include "tensor_basis.h"


namespace rpy {
namespace algebra {



extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraInterface<FreeTensor, TensorBasis>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensorInterface
    : public AlgebraInterface<FreeTensor, TensorBasis> {
public:
    using algebra_interface_t = AlgebraInterface<FreeTensor, TensorBasis>;

    using algebra_interface_t::algebra_interface_t;

    RPY_NO_DISCARD
    virtual FreeTensor exp() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensor log() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensor inverse() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensor antipode() const = 0;
    virtual void fmexp(const FreeTensor &other) = 0;
};

template <typename, template <typename> class>
class FreeTensorImplementation;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensor
    : public AlgebraBase<FreeTensorInterface, FreeTensorImplementation> {
    using base_t = AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

public:
    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensor;

    RPY_NO_DISCARD
    FreeTensor exp() const;
    RPY_NO_DISCARD
    FreeTensor log() const;
    RPY_NO_DISCARD
    FreeTensor inverse() const;
    RPY_NO_DISCARD
    FreeTensor antipode() const;
    FreeTensor &fmexp(const FreeTensor &other);

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(FreeTensor) {
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

extern template class ROUGHPY_ALGEBRA_EXPORT BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensorBundleInterface
    : public BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor> {
public:
    using algebra_interface_t = BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;

    using algebra_interface_t::algebra_interface_t;

    RPY_NO_DISCARD
    virtual FreeTensorBundle exp() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensorBundle log() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensorBundle inverse() const = 0;
    RPY_NO_DISCARD
    virtual FreeTensorBundle antipode() const = 0;
    virtual void fmexp(const FreeTensorBundle &other) = 0;


};

template <typename, template <typename> class>
class FreeTensorBundleImplementation;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBundleBase<FreeTensorBundleInterface, FreeTensorBundleImplementation>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensorBundle
    : public AlgebraBundleBase<FreeTensorBundleInterface, FreeTensorBundleImplementation>
{
    using base_t = AlgebraBundleBase<FreeTensorBundleInterface, FreeTensorBundleImplementation>;

public:

    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensorBundle;

    RPY_NO_DISCARD
    FreeTensorBundle exp() const;
    RPY_NO_DISCARD
    FreeTensorBundle log() const;
    RPY_NO_DISCARD
    FreeTensorBundle inverse() const;
    RPY_NO_DISCARD
    FreeTensorBundle antipode() const;
    FreeTensorBundle &fmexp(const FreeTensorBundle &other);

    RPY_SERIAL_SERIALIZE_FN();
};


RPY_SERIAL_SERIALIZE_FN_IMPL(FreeTensorBundle) {
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

template <>
template <typename C>
typename FreeTensor::basis_type basis_setup_helper<FreeTensor>::get(const C &ctx) {
    return ctx.get_tensor_basis();
}

template <>
template <typename C>
typename FreeTensorBundle::basis_type basis_setup_helper<FreeTensorBundle>::get(const C &ctx) {
    return ctx.get_tensor_basis();
}

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::FreeTensor, rpy::serial::specialization::member_serialize)
RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::FreeTensorBundle, rpy::serial::specialization::member_serialize)

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_H_
