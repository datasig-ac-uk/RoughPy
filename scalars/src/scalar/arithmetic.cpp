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
#include "casts.h"
#include "do_macro.h"

#include <roughpy/core/alloc.h>
#include <roughpy/device/host_device.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_types.h>
#include <roughpy/scalars/scalar_type.h>
#include "traits.h"

using namespace rpy;
using namespace rpy::scalars;

using rpy::scalars::dtl::ScalarContentType;

static inline void
uminus_impl(void* dst, const void* ptr, const devices::TypeInfo& info) {}

Scalar Scalar::operator-() const
{
    Scalar result;
    const auto info = type_info();
    optional<const ScalarType*> stype;
    if (!fast_is_zero()) {
        switch (p_type_and_content_type.get_enumeration()) {
            case ScalarContentType::TrivialBytes:
            case ScalarContentType::ConstTrivialBytes: result.
                        p_type_and_content_type = type_pointer(
                            info,
                            dtl::ScalarContentType::TrivialBytes
                        );
                uminus_impl(result.trivial_bytes, trivial_bytes, info);
                break;
            case ScalarContentType::OpaquePointer:
            case ScalarContentType::ConstOpaquePointer:
            case ScalarContentType::OwnedPointer: stype = type();
                RPY_CHECK(stype);
                result.p_type_and_content_type = type_pointer(
                    *stype,
                    dtl::ScalarContentType::OwnedPointer
                );
                result.allocate_data();
                uminus_impl(result.opaque_pointer, opaque_pointer, info);
                break;
            case ScalarContentType::Interface:
            case ScalarContentType::OwnedInterface: result.
                        p_type_and_content_type = type_pointer(
                            *stype,
                            dtl::ScalarContentType::OwnedPointer
                        );
                result.allocate_data();
                uminus_impl(result.opaque_pointer, interface->pointer(), info);
                break;
        }
    }
    return result;
}

namespace {

struct RPY_LOCAL AddInplace
{
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept { lhs += rhs; }
};

struct RPY_LOCAL SubInplace
{
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept { lhs -= rhs; }
};

struct RPY_LOCAL MulInplace
{
    template <typename T>
    constexpr void operator()(T& lhs, const T& rhs) noexcept { lhs *= rhs; }
};

struct RPY_LOCAL DivInplace
{
    template <typename T, typename R>
    constexpr enable_if_t<
        !is_same<T, rational_poly_scalar>::value &&
        is_same<
        decltype(std::declval<T&>() /= std::declval<const R&>()), T&>::value>
    operator()(T& lhs, const R& rhs) noexcept { lhs /= rhs; }

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
        Scalar tmp(dst_type);
        void* tmp_ptr = tmp.mut_pointer();
        scalars::dtl::scalar_convert_copy(tmp_ptr, dst_info, src, src_info, 1);
        do_op(dst, tmp_ptr, dst_info, std::forward<Op>(op));
    }
}

}// namespace

Scalar& Scalar::operator+=(const Scalar& other)
{
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(
            mut_pointer(),
            p_type_and_content_type,
            other.pointer(),
            other.p_type_and_content_type,
            AddInplace()
        );
    }
    return *this;
}

Scalar& Scalar::operator-=(const Scalar& other)
{
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(
            mut_pointer(),
            p_type_and_content_type,
            other.pointer(),
            other.p_type_and_content_type,
            SubInplace()
        );
    }
    return *this;
}

Scalar& Scalar::operator*=(const Scalar& other)
{
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(
            mut_pointer(),
            p_type_and_content_type,
            other.pointer(),
            other.p_type_and_content_type,
            MulInplace()
        );
    } else { operator=(0); }

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
static inline
enable_if_t<is_same<decltype(std::declval<T&>() /= std::declval<const R&>()), T&>::value>
do_divide_impl(T* dst, const R* divisor)
{
    *dst /= *divisor;
}

static inline void do_divide_impl(rational_poly_scalar* dst, const rational_scalar_type* divisor)
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
static inline void do_divide(T* dst,
                             const void* divisor,
                             const devices::TypeInfo& info)
{
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

    type_pointer tmp_type;
    if (other.p_type_and_content_type.is_pointer()) {
        tmp_type = type_pointer(other.p_type_and_content_type->rational_type(),
                                other.p_type_and_content_type.
                                      get_enumeration());
    } else {
        tmp_type = type_pointer(
            traits::rational_type_of(
                other.p_type_and_content_type.get_type_info()),
            other.p_type_and_content_type.get_enumeration());
    }

    const auto num_type = type_info();
    if (tmp_type == other.p_type_and_content_type) {
#define X(TP) do_divide((TP*) mut_pointer(),\
        other.pointer(),\
        tmp_type.get_type_info());\
        break

        DO_FOR_EACH_X(num_type)
#undef X
    } else {
        Scalar tmp(tmp_type);
        // hopefully converts
        tmp = other;
#define X(TP) do_divide((TP*) mut_pointer(),\
        tmp.pointer(),\
        tmp_type.get_type_info());\
        break

        DO_FOR_EACH_X(num_type)
#undef X

    }

    return *this;
}
