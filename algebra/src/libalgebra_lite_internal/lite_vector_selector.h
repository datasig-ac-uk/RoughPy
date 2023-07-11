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

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H

#include <roughpy/algebra/algebra_fwd.h>

#include <libalgebra_lite/dense_vector.h>
#include <libalgebra_lite/free_tensor.h>
#include <libalgebra_lite/lie.h>
#include <libalgebra_lite/shuffle_tensor.h>
#include <libalgebra_lite/sparse_vector.h>
#include <libalgebra_lite/vector.h>

namespace rpy {
namespace algebra {
namespace dtl {

template <typename VT>
using storage_type = lal::dtl::standard_storage<VT>;

template <VectorType>
struct vector_type_selector;

template <>
struct vector_type_selector<VectorType::Dense> {

    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::dense_vector, storage_type>;

    template <typename C>
    using shuffle_tensor
            = lal::shuffle_tensor<C, lal::dense_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::dense_vector, storage_type>;
};

template <>
struct vector_type_selector<VectorType::Sparse> {

    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::sparse_vector, storage_type>;

    template <typename C>
    using shuffle_tensor
            = lal::shuffle_tensor<C, lal::sparse_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::sparse_vector, storage_type>;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H
