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

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H

#include <roughpy/algebra/algebra_iterator.h>
#include <roughpy/algebra/algebra_iterator_impl.h>

#include <libalgebra_lite/dense_vector.h>

namespace rpy {
namespace algebra {

template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_helper_trait<
        lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>> {
    using iter_t
            = lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>;

    static auto key(const iter_t& it) noexcept -> decltype(it->key())
    {
        return it->key();
    }
    static auto value(const iter_t& it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};

template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_helper_trait<
        lal::dtl::dense_vector_const_iterator<Basis, Coefficients, Iterator>> {
    using iter_t = lal::dtl::dense_vector_const_iterator<
            Basis, Coefficients, Iterator>;

    static auto key(const iter_t& it) noexcept -> decltype(it->key())
    {
        return it->key();
    }
    static auto value(const iter_t& it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H
