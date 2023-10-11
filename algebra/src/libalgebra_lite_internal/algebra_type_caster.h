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

//
// Created by user on 18/07/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_

#include "implementors/algebra_impl.h"
#include "implementors/free_tensor_impl.h"
#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/shuffle_tensor.h>

#include "algebra_type_helper.h"
#include "lite_vector_selector.h"
#include "vector_type_helper.h"

namespace rpy {
namespace algebra {

template <AlgebraType Type>
struct algebra_type_selector;

template <>
struct algebra_type_selector<AlgebraType::FreeTensor> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template free_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::ShuffleTensor> {
    template <typename C, VectorType V>
    using type =
            typename dtl::vector_type_selector<V>::template shuffle_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::Lie> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template lie<C>;
};
template <>
struct algebra_type_selector<AlgebraType::FreeTensorBundle> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template free_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::ShuffleTensorBundle> {
    template <typename C, VectorType V>
    using type =
            typename dtl::vector_type_selector<V>::template shuffle_tensor<C>;
};
template <>
struct algebra_type_selector<AlgebraType::LieBundle> {
    template <typename C, VectorType V>
    using type = typename dtl::vector_type_selector<V>::template lie<C>;
};

template <
        typename Coefficients, typename AlgebraTag = void,
        typename VectorTag = void>
struct algebra_type_caster;

template <typename Coefficients>
struct algebra_type_caster<Coefficients, void, void> {
    template <AlgebraType AType>
    using refine
            = algebra_type_caster<Coefficients, algebra_type_tag<AType>, void>;
};

template <typename Coefficients, typename ATag>
struct algebra_type_caster<Coefficients, ATag, void> {
    template <VectorType VType>
    using refine
            = algebra_type_caster<Coefficients, ATag, vector_type_tag<VType>>;
};

template <typename Coefficients, AlgebraType AType, VectorType VType>
struct algebra_type_caster<
        Coefficients, algebra_type_tag<AType>, vector_type_tag<VType>> {

    using type = typename algebra_type_selector<AType>::template type<
            Coefficients, VType>;
    using info = dtl::alg_details_of<type>;
    using interface_t = typename info::interface_type;

    static type& cast(RawUnspecifiedAlgebraType arg)
    {
        RPY_CHECK(arg != nullptr);
        RPY_CHECK(arg->alg_type() == AType);
        RPY_CHECK(arg->storage_type() == VType);
        auto* interface_ptr = reinterpret_cast<interface_t*>(arg);
        return algebra_cast<type>(*interface_ptr);
    }
    static const type& cast(ConstRawUnspecifiedAlgebraType arg) {
        RPY_CHECK(arg != nullptr);
        RPY_CHECK(arg->alg_type() == AType);
        RPY_CHECK(arg->storage_type() == VType);
        const auto* interface_ptr = reinterpret_cast<const interface_t*>(arg);
        return algebra_cast<type>(*interface_ptr);
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_CASTER_H_
