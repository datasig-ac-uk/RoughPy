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

//
// Created by user on 02/04/23.
//

#ifndef ROUGHPY_LA_CONTEXT_INCLUDE_ROUGHPY_LA_CONTEXT_VECTOR_TYPE_SELECTOR_H
#define ROUGHPY_LA_CONTEXT_INCLUDE_ROUGHPY_LA_CONTEXT_VECTOR_TYPE_SELECTOR_H

#include <roughpy/algebra/algebra_fwd.h>

#include <libalgebra/dense_vector.h>
#include <libalgebra/lie.h>
#include <libalgebra/sparse_vector.h>
#include <libalgebra/tensor.h>
#include <libalgebra/vector.h>

namespace rpy {
namespace algebra {
namespace dtl {

template <VectorType VType>
struct LAVectorSelector;

template <>
struct LAVectorSelector<VectorType::Dense> {
    template <deg_t W, deg_t D, typename C>
    using lie_t = alg::lie<C, W, D, alg::vectors::dense_vector>;

    template <deg_t W, deg_t D, typename C>
    using ftensor_t
            = alg::free_tensor<C, W, D, alg::vectors::dense_vector,
                               alg::traditional_free_tensor_multiplication>;

    template <deg_t W, deg_t D, typename C>
    using stensor_t
            = alg::shuffle_tensor<C, W, D /*, alg::vectors::dense_vector*/>;
};

template <>
struct LAVectorSelector<VectorType::Sparse> {
    template <deg_t W, deg_t D, typename C>
    using lie_t = alg::lie<C, W, D, alg::vectors::sparse_vector>;

    template <deg_t W, deg_t D, typename C>
    using ftensor_t
            = alg::free_tensor<C, W, D, alg::vectors::sparse_vector,
                               alg::traditional_free_tensor_multiplication>;

    template <deg_t W, deg_t D, typename C>
    using stensor_t
            = alg::shuffle_tensor<C, W, D /*, alg::vectors::sparse_vector*/>;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_LA_CONTEXT_INCLUDE_ROUGHPY_LA_CONTEXT_VECTOR_TYPE_SELECTOR_H
