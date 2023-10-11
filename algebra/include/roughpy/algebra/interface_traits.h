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

#ifndef ROUGHPY_ALGEBRA_INTERFACE_TRAITS_H_
#define ROUGHPY_ALGEBRA_INTERFACE_TRAITS_H_

namespace rpy {
namespace algebra {

namespace traits {

namespace dtl {

template <typename I>
struct basis_of_impl {
    using type = typename I::basis_t;
};

template <typename I>
struct key_of_impl {
    using type = typename basis_of_impl<I>::type::key_type;
};

template <typename I>
struct algebra_of_impl {
    using type = typename I::algebra_t;
};

}

template <typename I>
using basis_of = typename dtl::basis_of_impl<I>::type;

template <typename I>
using key_of = typename dtl::key_of_impl<I>::type;

template <typename I>
using algebra_of = typename dtl::algebra_of_impl<I>::type;

template <typename BI>
using base_algebra_of = typename dtl::algebra_of_impl<BI>::type;

template <typename BI>
using fibre_algebra_of = typename dtl::algebra_of_impl<BI>::type;

}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_INTERFACE_TRAITS_H_
