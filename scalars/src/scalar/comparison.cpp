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
// Created by user on 01/11/23.
//

#include "comparison.h"

#include "do_macro.h"
#include <roughpy/scalars/scalar_types.h>

#include <limits>

using namespace rpy;
using namespace rpy::scalars;

using devices::TypeInfo;

namespace {

template <typename T>
constexpr bool is_zero(const T& arg) noexcept
{
    return arg == T(0);
}

constexpr bool is_zero(const rational_scalar_type& arg) noexcept
{
    return arg == 0;
}

inline bool is_zero(const rational_poly_scalar& arg) noexcept
{
    return arg.empty();
}

template <typename L, typename R>
inline bool compare_outer(const L& lhs, const R& rhs) noexcept;

template <typename S, typename T>
constexpr bool compare_impl(const S& lhs, const T& rhs) noexcept
{
    return lhs == rhs;
}

RPY_UNUSED
inline bool
compare_same(const void* lhs, const void* rhs, TypeInfo info) noexcept
{
#define X(T) return compare_impl(*((const T*) lhs), *((const T*) rhs))
    DO_FOR_EACH_X(info)
#undef X
    return false;
}

template <typename T>
constexpr bool compare_wrap_equal(const T& lhs, const T& rhs) noexcept
{
    return lhs == rhs;
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_integral<R>::value
                && is_signed<L>::value && is_unsigned<R>::value,
        bool>
compare_wrap_equal(const L& lhs, const R& rhs) noexcept
{
    static_assert(sizeof(L) == sizeof(R), "");
    return (lhs >= 0) && static_cast<R>(lhs) == rhs;
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_integral<R>::value
                && is_unsigned<L>::value && is_signed<R>::value,
        bool>
compare_wrap_equal(const L& lhs, const R& rhs) noexcept
{
    static_assert(sizeof(L) == sizeof(R), "");
    return (rhs >= 0) && lhs == static_cast<R>(rhs);
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_signed<L>::value
                && is_floating_point<R>::value,
        bool>
compare_wrap_equal(const L& lhs, const R& rhs) noexcept
{
    /*
     * Left is signed integral, right is fp, make sure lhs is smaller than the
     * mantissa bits of rhs
     */
    constexpr L unique_max = (L(1) << std::numeric_limits<R>::digits) - 1;
    return (lhs <= unique_max && lhs >= -unique_max)
            && static_cast<R>(lhs) == rhs;
}
template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_unsigned<L>::value
                && is_floating_point<R>::value,
        bool>
compare_wrap_equal(const L& lhs, const R& rhs) noexcept
{
    /*
     * Left is unsigned integral, right is fp, make sure lhs is smaller than the
     * mantissa bits of rhs
     */
    constexpr L unique_max = (L(1) << std::numeric_limits<R>::digits) - 1;
    return (lhs <= unique_max) && static_cast<R>(lhs) == rhs;
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_floating_point<L>::value
                && is_integral<R>::value,
        bool>
compare_wrap_equal(const L& lhs, const R& rhs) noexcept
{
    // Just reverse and compare.
    return compare_wrap_equal(rhs, lhs);
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_signed<L>::value
                && is_integral<R>::value && is_unsigned<R>::value,
        bool>
compare_wrap_unequal(const L& lhs, const R& rhs) noexcept
{
    return (lhs >= 0) && static_cast<R>(lhs) == rhs;
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value && is_unsigned<L>::value
                && is_integral<R>::value && is_signed<R>::value,
        bool>
compare_wrap_unequal(const L& lhs, const R& rhs) noexcept
{
    return (rhs >= 0) && static_cast<R>(lhs) == rhs;
}

template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_integral<L>::value
                && is_floating_point<R>::value,
        bool>
compare_wrap_unequal(const L& lhs, const R& rhs) noexcept
{
    return static_cast<R>(lhs) == rhs;
}
template <typename L, typename R>
constexpr enable_if_t<
        !is_same<L, R>::value && is_floating_point<L>::value
                && is_integral<R>::value,
        bool>
compare_wrap_unequal(const L& lhs, const R& rhs) noexcept
{
    // Just reverse and compare.
    return compare_wrap_unequal(rhs, lhs);
}

template <typename L>
constexpr bool
compare_wrap_unequal(const L& lhs, const rational_scalar_type& rhs) noexcept
{
    return rational_scalar_type(lhs) == rhs;
}
template <typename R>
constexpr enable_if_t<
        sizeof(rational_scalar_type) < sizeof(R)
                && !is_same<R, rational_poly_scalar>::value,
        bool>
compare_wrap_unequal(const rational_scalar_type& lhs, const R& rhs) noexcept
{
    return lhs == rational_scalar_type(rhs);
}
template <typename L>
inline  bool
compare_wrap_unequal(const L& lhs, const rational_poly_scalar& rhs) noexcept
{
    if (rhs.empty()) { return is_zero(lhs); }

    if (rhs.size() == 1) {
        auto first = rhs.begin();
        if (first->key() == monomial()) {
            return compare_outer(lhs, first->value());
        }
    }

    return false;
}
template <typename R>
constexpr bool
compare_wrap_unequal(const rational_poly_scalar& lhs, const R& rhs) noexcept
{
    return compare_wrap_unequal(rhs, lhs);
}

template <typename L, typename R>
constexpr enable_if_t<sizeof(L) == sizeof(R), bool>
compare_wrap(const L& lhs, const R& rhs) noexcept
{
    return compare_wrap_equal(lhs, rhs);
}

template <typename L, typename R>
constexpr enable_if_t<sizeof(L) < sizeof(R), bool>
compare_wrap(const L& lhs, const R& rhs) noexcept
{
    return compare_wrap_unequal(lhs, rhs);
}

template <typename L, typename R>
constexpr enable_if_t<sizeof(R) < sizeof(L), bool>
compare_wrap(const L& lhs, const R& rhs) noexcept
{
    return compare_wrap_unequal(rhs, lhs);
}

/*
 * Since half and bfloat16 won't correctly identify as floating types, we
 * replace them with floats. Everything else gets passed through without
 * modification.
 */

template <typename T>
constexpr const T& replace_tricky(const T& arg) noexcept
{
    return arg;
}

inline float replace_tricky(const half& arg) noexcept { return float(arg); }
inline float replace_tricky(const bfloat16& arg) noexcept
{
    return float(arg);
}
template <typename L, typename R>
inline bool compare_outer(const L& lhs, const R& rhs) noexcept
{
    return compare_wrap(replace_tricky(lhs), replace_tricky(rhs));
}

template <typename L>
inline bool
compare_set_left(const L& lhs, const void* rhs, TypeInfo rhs_info) noexcept
{
#define X(T) return compare_outer(lhs, *((const T*) rhs))
    DO_FOR_EACH_X(rhs_info)
#undef X
    return false;
}

inline bool
eq_impl(const void* lhs, TypeInfo lhs_info, const void* rhs, TypeInfo rhs_info
) noexcept
{

#define X(T) return compare_set_left(*((const T*) lhs), rhs, rhs_info)
    DO_FOR_EACH_X(lhs_info)
#undef X
    return false;
}

}// namespace

bool scalars::dtl::is_pointer_zero(
        const void* ptr,
        const PackedScalarTypePointer<scalars::dtl::ScalarContentType>& p_type
) noexcept
{
    const auto info = (p_type.is_pointer() ? p_type.get_pointer()->type_info() : p_type.get_type_info());

#define X(TYP) return is_zero(*((const TYP*) ptr))
    DO_FOR_EACH_X(info)
#undef X
    return true;
}

bool Scalar::operator==(const Scalar& other) const
{
    if (fast_is_zero()) { return other.is_zero(); }
    if (other.fast_is_zero()) { return is_zero(); }

    return eq_impl(pointer(), type_info(), other.pointer(), other.type_info());
}
