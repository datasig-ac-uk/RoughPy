//
// Created by user on 03/03/23.
//

#ifndef ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H
#define ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H

#include <type_traits>

#include <boost/call_traits.hpp>
#include <boost/type_traits/copy_cv.hpp>
#include <boost/type_traits/copy_cv_ref.hpp>
#include <boost/type_traits/detected.hpp>
#include <boost/type_traits/detected_or.hpp>
#include <boost/type_traits/is_detected.hpp>
#include <boost/type_traits/remove_cv_ref.hpp>

namespace rpy {
namespace traits {

using std::declval;

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


}// namespace traits
}// namespace rpy

#endif//ROUGHPY_CORE_INCLUDE_ROUGHPY_CORE_TRAITS_H
