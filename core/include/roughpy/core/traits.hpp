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


#include <type_traits>
#include <utility>

#include <boost/type_traits/copy_cv.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <boost/type_traits/is_detected.hpp>
#include <boost/type_traits/remove_cv_ref.hpp>
#include <boost/type_traits/type_identity.hpp>

#include "macros.h"

namespace rpy {

using std::declval;

// Base classes for traits
using std::integral_constant;
using std::true_type;
using std::false_type;

// Primary type categories
using std::is_void_v;
using std::is_null_pointer_v;
using std::is_integral_v;
using std::is_floating_point_v;
using std::is_array_v;
using std::is_enum_v;
using std::is_union_v;
using std::is_class_v;
using std::is_function_v;
using std::is_pointer_v;
using std::is_lvalue_reference_v;
using std::is_rvalue_reference_v;
using std::is_member_object_pointer_v;
using std::is_member_function_pointer_v;

// Composite type categories
using std::is_fundamental_v;
using std::is_arithmetic_v;
using std::is_scalar_v;
using std::is_object_v;
using std::is_compound_v;
using std::is_reference_v;
using std::is_member_pointer_v;

// Type properties
using std::is_const_v;
using std::is_volatile_v;
using std::is_trivial_v;
using std::is_trivially_copyable_v;
using std::is_standard_layout_v;
// using std::is_pod_v; // deprecated
// using std::is_literal_type_v; // deprecated
using std::is_empty_v;
using std::is_polymorphic_v;
using std::is_abstract_v;
using std::is_final_v;
using std::is_aggregate_v;
using std::is_signed_v;
using std::is_unsigned_v;
// using std::is_bounded_array_v; // C++20
// using std::is_unbounded_array_v; // C++20

// Supported operations
using std::is_constructible_v;
using std::is_trivially_constructible_v;
using std::is_nothrow_constructible_v;
using std::is_default_constructible_v;
using std::is_trivially_default_constructible_v;
using std::is_nothrow_default_constructible_v;
using std::is_copy_constructible_v;
using std::is_trivially_copy_constructible_v;
using std::is_nothrow_copy_constructible_v;
using std::is_move_constructible_v;
using std::is_trivially_move_constructible_v;
using std::is_nothrow_move_constructible_v;
using std::is_assignable_v;
using std::is_trivially_assignable_v;
using std::is_nothrow_assignable_v;
using std::is_copy_assignable_v;
using std::is_trivially_copy_assignable_v;
using std::is_nothrow_copy_assignable_v;
using std::is_move_assignable_v;
using std::is_trivially_move_assignable_v;
using std::is_nothrow_move_assignable_v;
using std::is_destructible_v;
using std::is_trivially_destructible_v;
using std::is_nothrow_destructible_v;
using std::has_virtual_destructor_v;
using std::is_swappable_v;
using std::is_swappable_with_v;
using std::is_nothrow_swappable_v;
using std::is_nothrow_swappable_with_v;

// Property queries
using std::alignment_of_v;
using std::rank_v;
using std::extent_v;

// Type relationships
using std::is_same_v;
using std::is_base_of_v;
using std::is_convertible_v;
// using std::is_nothrow_convertible_v; // C++20
using std::is_trivially_constructible_v;
using std::is_nothrow_constructible_v;
using std::is_assignable_v;
using std::is_trivially_assignable_v;
using std::is_nothrow_assignable_v;
using std::is_invocable_v;
using std::is_invocable_r_v;
using std::is_nothrow_invocable_v;
using std::is_nothrow_invocable_r_v;

// Const-volatile modifications
using std::remove_cv_t;
using std::remove_const_t;
using std::remove_volatile_t;
using std::add_cv_t;
using std::add_const_t;
using std::add_volatile_t;

// Reference modifications
using std::remove_reference_t;
using std::add_lvalue_reference_t;
using std::add_rvalue_reference_t;

// Pointer modifications
using std::remove_pointer_t;
using std::add_pointer_t;

// Sign modifications
using std::make_signed_t;
using std::make_unsigned_t;

// Array modifications
using std::remove_extent_t;
using std::remove_all_extents_t;

// Miscellaneous transformations
// using std::aligned_storage_t; // deprecated
// using std::aligned_union_t; // deprecated
using std::decay_t;
using std::enable_if_t;
using std::conditional_t;
using std::common_type_t;
using std::underlying_type_t;
// using std::result_of_t; // deprecated
using std::invoke_result_t;
using std::void_t;

// remove_cvref_t is C++20, using boost instead
template <typename T>
using remove_cvref_t = boost::remove_cv_ref_t<T>;

using boost::type_identity_t;

// Copy cv
using boost::copy_cv_t;

template <typename T, typename U>
using copy_cvref_t = boost::copy_cv_ref_t<T, U>;



using boost::is_detected_v;


// Integer sequences
using std::integer_sequence;
using std::make_integer_sequence;

template <typename... T>
using integer_sequence_for = make_integer_sequence<std::size_t, sizeof...(T)>;


// Helper tags
using std::piecewise_construct_t;
using std::piecewise_construct;
using std::in_place_t;
using std::in_place;

// Probably not used;
// using std::in_place_index_t;
// using std::in_place_index;
// using std::in_place_type_t;
// using std::in_place_type;
// using std::nontype_t;
// using std::nontype;


/**
 * @brief Ensure that the type T is a pointer.
 *
 * Makes T a pointer if it isn't already a pointer.
 */
template <typename T>
using ensure_pointer = conditional_t<is_pointer_v<T>, T, add_pointer_t<T>>;



struct EmptyType {};

template <typename T, typename B = EmptyType>
using void_or_base = conditional_t<is_void_v<T>, B, T>;


template <size_t N>
struct ConstLog2 : integral_constant<size_t, ConstLog2<N / 2>::value + 1>{ };
template <>
struct ConstLog2<1> : integral_constant<size_t, 0>{};



template <typename... Ts>
RPY_INLINE_ALWAYS void ignore_unused(Ts&&...)
{}


}// namespace rpy

#endif// ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H
