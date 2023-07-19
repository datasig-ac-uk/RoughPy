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
// Created by user on 03/03/23.
//

#ifndef ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H
#define ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H

#include "macros.h"

#include <type_traits>

#include <boost/call_traits.hpp>

#include <boost/type_traits/copy_cv.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <boost/type_traits/detected.hpp>
#include <boost/type_traits/detected_or.hpp>
#include <boost/type_traits/is_detected.hpp>
#include <boost/type_traits/remove_cv_ref.hpp>

#include <boost/container_hash/hash.hpp>

namespace rpy {

using std::declval;
using std::integral_constant;
using std::move;
using std::forward;

using std::is_array;
using std::is_class;
using std::is_enum;
using std::is_floating_point;
using std::is_function;
using std::is_integral;
using std::is_lvalue_reference;
using std::is_member_function_pointer;
using std::is_member_object_pointer;
using std::is_null_pointer;
using std::is_pointer;
using std::is_rvalue_reference;
using std::is_union;
using std::is_void;

using std::is_arithmetic;
using std::is_compound;
using std::is_fundamental;
using std::is_member_pointer;
using std::is_object;
using std::is_reference;
using std::is_scalar;

using std::is_abstract;
using std::is_const;
using std::is_empty;
using std::is_final;
using std::is_polymorphic;
using std::is_signed;
using std::is_standard_layout;
using std::is_trivial;
using std::is_trivially_copyable;
using std::is_unsigned;
using std::is_volatile;

using std::is_constructible;
using std::is_default_constructible;
using std::is_nothrow_constructible;
using std::is_nothrow_default_constructible;
using std::is_trivially_constructible;
using std::is_trivially_default_constructible;

using std::is_base_of;
using std::is_convertible;
using std::is_same;

using std::add_const_t;
using std::add_cv_t;
using std::add_volatile_t;
using std::remove_const_t;
using std::remove_cv_t;
using std::remove_volatile_t;

using std::add_lvalue_reference_t;
using std::add_pointer_t;
using std::add_rvalue_reference_t;
using std::make_signed_t;
using std::make_unsigned_t;
using std::remove_all_extents_t;
using std::remove_extent_t;
using std::remove_pointer_t;
using std::remove_reference_t;

using std::common_type;
using std::conditional_t;
using std::decay;
using std::enable_if_t;
using std::underlying_type_t;

using boost::copy_cv_ref_t;
using boost::copy_cv_t;
using boost::detected_or_t;
using boost::detected_t;
using boost::remove_cv_ref_t;

using boost::is_detected;

#ifdef RPY_CPP_17
using std::void_t;
using std::invoke_result_t;
#else
using boost::void_t;

template <typename F, typename... ArgTypes>
using invoke_result_t = std::result_of_t<F(ArgTypes...)>;
#endif

/**
 * @brief Ensure that the type T is a pointer.
 *
 * Makes T a pointer if it isn't already a pointer.
 */
template <typename T>
using ensure_pointer = conditional_t<is_pointer<T>::value, T, add_pointer_t<T>>;

/**
 * @brief Get the most sensible parameter type for type T
 */
template <typename T>
using param_type_t = typename boost::call_traits<T>::param_type;

namespace dtl {
template <typename... Ts>
struct select_first_impl;

template <typename First, typename... Ts>
struct select_first_impl<First, Ts...> {
    using type = First;
};

}// namespace dtl

template <typename... Ts>
using select_first_t = typename dtl::select_first_impl<Ts...>::type;

namespace dtl {
struct EmptyBase {
};
}// namespace dtl

template <typename T, typename B = dtl::EmptyBase>
using void_or_base = conditional_t<is_void<T>::value, B, T>;

using boost::hash;

}// namespace rpy

#endif// ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H
