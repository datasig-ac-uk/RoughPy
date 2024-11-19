// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by user on 01/11/23.
//

#include "arithmetic.h"

#include <roughpy/core/check.h>
#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/macros.h>

#include "traits.h"
#include <roughpy/platform/devices/host_device.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_types.h>

#include "do_macro.h"
#include "type_promotion.h"



using namespace rpy;
using namespace rpy::scalars;

using rpy::scalars::dtl::ScalarContentType;

static inline void
uminus_impl(void* dst, const void* ptr, const devices::TypeInfo& info)
{}

Scalar Scalar::operator-() const
{
    Scalar result;
    const auto info = type_info();
    optional<const ScalarType*> stype;
    if (!fast_is_zero()) {
        switch (p_type_and_content_type.get_enumeration()) {
            case ScalarContentType::TrivialBytes:
            case ScalarContentType::ConstTrivialBytes:
                result.p_type_and_content_type = type_pointer(
                        info,
                        dtl::ScalarContentType::TrivialBytes
                );
                uminus_impl(result.trivial_bytes, trivial_bytes, info);
                break;
            case ScalarContentType::OpaquePointer:
            case ScalarContentType::ConstOpaquePointer:
            case ScalarContentType::OwnedPointer:
                stype = type();
                RPY_CHECK(stype);
                result.p_type_and_content_type = type_pointer(
                        *stype,
                        dtl::ScalarContentType::OwnedPointer
                );
                result.allocate_data();
                uminus_impl(result.opaque_pointer, opaque_pointer, info);
                break;
            case ScalarContentType::Interface:
            case ScalarContentType::OwnedInterface:
                result.p_type_and_content_type = type_pointer(
                        *stype,
                        dtl::ScalarContentType::OwnedPointer
                );
                result.allocate_data();
                uminus_impl(
                        result.opaque_pointer,
                        interface_ptr->pointer(),
                        info
                );
                break;
        }
    }
    return result;
}

namespace {

struct RPY_LOCAL AddInplace {
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept
    {
        lhs += rhs;
    }
};

struct RPY_LOCAL SubInplace {
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept
    {
        lhs -= rhs;
    }
};

struct RPY_LOCAL MulInplace {
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept
    {
        lhs *= rhs;
    }
};

struct RPY_LOCAL DivInplace {
    template <typename T, typename R>
    constexpr enable_if_t<
            !is_same_v<T, rational_poly_scalar>
            && is_same_v<
                    decltype(std::declval<T&>() /= std::declval<const R&>()),
                    T&>>
    operator()(T& lhs, const R& rhs) noexcept
    {
        lhs /= rhs;
    }

    void operator()(...) { RPY_THROW(std::domain_error, "invalid division"); }
};

template <typename Op>
static inline void
do_op(void* dst, const void* src, devices::TypeInfo type, Op&& op)
{
#define X(T) return op(*((T*) dst), *((const T*) src))
    DO_FOR_EACH_X(type)
#undef X
    RPY_THROW(std::domain_error, "unsupported operation");
}

template <typename Op>
inline void scalar_inplace_arithmetic(
        void* dst,
        PackedScalarTypePointer<ScalarContentType> dst_type,
        const void* src,
        PackedScalarTypePointer<ScalarContentType> src_type,
        Op&& op
)
{
    auto src_info = type_info_from(src_type);
    auto dst_info = type_info_from(dst_type);

    if (src_info == dst_info) {
        do_op(dst, src, dst_info, std::forward<Op>(op));
    } else {
        Scalar tmp(dst_info);
        void* tmp_ptr = tmp.mut_pointer();
        scalars::dtl::scalar_convert_copy(tmp_ptr, dst_info, src, src_info, 1);
        do_op(dst, tmp_ptr, dst_info, std::forward<Op>(op));
    }
}

template <typename Op>
inline void scalar_inplace_arithmetic(Scalar& dst, const Scalar& src, Op&& op)
{
    const auto dst_info = dst.type_info();
    const auto src_info = src.type_info();

    if (src_info == dst_info) {
        do_op(dst.mut_pointer(), src.pointer(), dst_info, std::forward<Op>(op));
    } else {
        auto out_type = scalars::dtl::compute_dest_type(
                dst.packed_type_info(),
                src.packed_type_info()
        );
        if (out_type != dst_info) { dst.change_type(out_type); }
        scalar_inplace_arithmetic(
                dst.mut_pointer(),
                dst.packed_type_info(),
                src.pointer(),
                src.packed_type_info(),
                std::forward<Op>(op)
        );
    }
}

}// namespace

Scalar& Scalar::operator+=(const Scalar& other)
{
    RPY_DBG_ASSERT(!p_type_and_content_type.is_null());
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(*this, other, AddInplace());
    }
    return *this;
}

Scalar& Scalar::operator-=(const Scalar& other)
{
    RPY_DBG_ASSERT(!p_type_and_content_type.is_null());
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(*this, other, SubInplace());
    }
    return *this;
}

Scalar& Scalar::operator*=(const Scalar& other)
{
    RPY_DBG_ASSERT(!p_type_and_content_type.is_null());
    if (!fast_is_zero() && !other.fast_is_zero()) {
        scalar_inplace_arithmetic(*this, other, MulInplace());
    } else {
        operator=(0);
    }

    return *this;
}

/*
 * The stupid implicit conversion of C++ keeps kicking me while I'm down here.
 * Basically, the implicit conversion from half -> rational seems to pass muster
 * with the template argument deduction, only to fail at a later step because
 * there is no valid conversion from half->rational (and similar).
 *
 * For this reason, we do the type checking whilst the arguments are pointers,
 * to eliminate these as problems.
 */
template <typename T, typename R>
static inline enable_if_t<
        is_same_v<decltype(std::declval<T&>() /= std::declval<const R&>()),
                T&>>
do_divide_impl(T* dst, const R* divisor)
{
    *dst /= *divisor;
}

static inline void
do_divide_impl(rational_poly_scalar* dst, const rational_scalar_type* divisor)
{
    *dst /= *divisor;
}

template <typename R>
static inline void do_divide_impl(rational_poly_scalar* dst, const R* other)
{
    RPY_THROW(std::domain_error, "invalid division");
}

static inline void do_divide_impl(...)
{
    RPY_THROW(std::domain_error, "invalid division");
}

template <typename T>
static inline void
do_divide(T* dst, const void* divisor, const devices::TypeInfo& info){
#define X(TP) return do_divide_impl(dst, (const TP*) divisor)
        DO_FOR_EACH_X(info)
#undef X
}

Scalar& Scalar::operator/=(const Scalar& other)
{
    if (other.fast_is_zero()) {
        RPY_THROW(std::domain_error, "division by zero");
    }

    /*
     * We need to check some additional things here.
     * Division need not be defined for all scalar types. For instance, integers
     * do not have globally defined division (it is a ring, not a field).
     * Similarly, the type of divisible units might differ from the ring/field
     * type - as is the case for polynomials.
     * We need to handle both of these cases gracefully.
     */
    const auto num_info = type_info_from(p_type_and_content_type);
    const auto true_denom_info = type_info_from(other.p_type_and_content_type);
    const auto rat_denom_info = traits::rational_type_of(true_denom_info);

    // Set up the actual denominator that we will divide by
    Scalar rational_denom(true_denom_info, other.pointer());
    if (rat_denom_info != true_denom_info) {
        rational_denom.change_type(rat_denom_info);
    }

    /*
     * The type resolution needs to be done with the true denominator type,
     * rather than whatever the rational type is.
     */
    if (num_info != true_denom_info) {
        change_type(dtl::compute_dest_type(
                p_type_and_content_type,
                other.p_type_and_content_type
        ));
    }

    // Now we should be ready
#define X(tp) do_divide((tp*) mut_pointer(), rational_denom.pointer(), rat_denom_info);\
    break
    DO_FOR_EACH_X(num_info)
#undef X

    return *this;
}
