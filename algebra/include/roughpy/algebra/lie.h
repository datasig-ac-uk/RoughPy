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

#ifndef ROUGHPY_ALGEBRA_LIE_H_
#define ROUGHPY_ALGEBRA_LIE_H_

#include "algebra_base.h"
#include "algebra_bundle.h"

#include "lie_basis.h"


namespace rpy {
namespace algebra {


extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraInterface<Lie, LieBasis>;

using LieInterface = AlgebraInterface<Lie, LieBasis>;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBase<LieInterface>;

class ROUGHPY_ALGEBRA_EXPORT Lie : public AlgebraBase<LieInterface> {
    using base_t = AlgebraBase<LieInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::Lie;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();

};

RPY_SERIAL_SERIALIZE_FN_IMPL(Lie) {
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

class LieBundle;

extern template class ROUGHPY_ALGEBRA_EXPORT BundleInterface<LieBundle, Lie, Lie>;

using LieBundleInterface = BundleInterface<LieBundle, Lie, Lie>;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBundleBase<LieBundleInterface>;

class ROUGHPY_ALGEBRA_EXPORT LieBundle : public AlgebraBundleBase<LieBundleInterface> {
    using base_t = AlgebraBundleBase<LieBundleInterface>;
public:

    static constexpr AlgebraType s_alg_type = AlgebraType::LieBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(LieBundle) {
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

template <>
template <typename C>
typename Lie::basis_type basis_setup_helper<Lie>::get(const C& ctx) {
    return ctx.get_lie_basis();
}

template <>
template <typename C>
typename Lie::basis_type basis_setup_helper<LieBundle>::get(const C& ctx) {
    return ctx.get_lie_basis();
}


}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::Lie, rpy::serial::specialization::member_serialize)
RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::LieBundle, rpy::serial::specialization::member_serialize)

#endif// ROUGHPY_ALGEBRA_LIE_H_
