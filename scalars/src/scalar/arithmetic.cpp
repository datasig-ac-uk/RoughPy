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

#include "arithmetic.h"
#include "casts.h"

#include <roughpy/core/alloc.h>
#include <roughpy/device/host_device.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_types.h>

using namespace rpy;
using namespace rpy::scalars;

using rpy::scalars::dtl::ScalarContentType;

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
    constexpr void operator()(T& lhs, const R& rhs) noexcept
    {
        lhs /= rhs;
    }
};

#define X(T) return op(*((T*) dst), *((const T*) src))

template <typename Op>
static inline void
do_op(void* dst, const void* src, devices::TypeInfo type, Op&& op)
{
    switch (type.code) {
        case devices::TypeCode::Int:
            switch (type.bytes) {
                case 1: X(int8_t);
                case 2: X(int16_t);
                case 4: X(int32_t);
                case 8: X(int64_t);
            }
            break;
        case devices::TypeCode::UInt:
            switch (type.bytes) {
                case 1: X(uint8_t);
                case 2: X(uint16_t);
                case 4: X(uint32_t);
                case 8: X(uint64_t);
            }
            break;
        case devices::TypeCode::Float:
            switch (type.bytes) {
                case 2: X(half);
                case 4: X(float);
                case 8: X(double);
            }
            break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat:
            if (type.bytes == 2) { X(bfloat16); }
            break;
        case devices::TypeCode::Complex: break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::ArbitraryPrecision: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionRational:
            X(rational_scalar_type);
        case devices::TypeCode::APRationalPolynomial: X(rational_poly_scalar);
    }
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
    auto src_info = src_type.get_type_info();
    auto dst_info = dst_type.get_type_info();

    if (src_info == dst_info) {
        do_op(dst, src, dst_info, std::forward<Op>(op));
    } else {
        Scalar tmp(dst_type);
        void* tmp_ptr = tmp.mut_pointer();
        scalars::dtl::scalar_convert_copy(tmp_ptr, dst_info, src, src_info);
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
Scalar& Scalar::operator-=(const Scalar& other) {
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
Scalar& Scalar::operator*=(const Scalar& other) {
    if (!other.fast_is_zero()) {
        scalar_inplace_arithmetic(
                mut_pointer(),
                p_type_and_content_type,
                other.pointer(),
                other.p_type_and_content_type,
                MulInplace()
        );
    } else {
        operator=(0);
    }

    return *this;
}
Scalar& Scalar::operator/=(const Scalar& other)
{
    if (other.fast_is_zero()) {
        RPY_THROW(std::domain_error, "division by zero");
    }

    return *this;
}
