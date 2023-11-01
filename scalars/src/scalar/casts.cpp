

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

#include <roughpy/scalars/scalar.h>

#include <roughpy/core/alloc.h>
#include <roughpy/scalars/scalar_types.h>
#include "casts.h"

using namespace rpy;
using namespace scalars;
using devices::TypeInfo;

template <typename D, typename T>
static inline enable_if_t<
        is_constructible<D, const T&>::value
                && is_trivially_default_constructible<D>::value,
        bool>
write_result(D* dst, const T& src) noexcept
{
    construct_inplace(dst, src);
    return true;
}

template <typename D, typename T>
static inline enable_if_t<
        is_constructible<D, const T&>::value
                && !is_trivially_default_constructible<D>::value,
        bool>
write_result(D* dst, const T& src) noexcept
{
    try {
        *dst = D(src);
    } catch (...) {
        return false;
    }
    return true;
}

template <typename D, typename T>
static inline enable_if_t<!is_constructible<D, const T&>::value, bool>
write_result(D*, const T&) noexcept
{
    return false;
}

template <typename D>
static inline bool write_result(D* dst, const rational_poly_scalar& value)
        noexcept {
    if (value.empty()) {
        write_result(dst, 0);
        return true;
    }
    if (value.size() == 1) {
        auto kv = value.begin();
        if (kv->key() == monomial()) {
            // Try converting via rational
            return write_result(dst, kv->value());
        }
    }
    return false;
}


template <typename T>
static inline bool
convert_impl(void* dst, const TypeInfo& info, const T& src) noexcept
{
    switch (info.code) {
        case devices::TypeCode::Int:
            switch (info.bytes) {
                case 1: return write_result((int8_t*) dst, src);
                case 2: return write_result((int16_t*) dst, src);
                case 4: return write_result((int32_t*) dst, src);
                case 8: return write_result((int64_t*) dst, src);
            }
            break;
        case devices::TypeCode::UInt:
            switch (info.bytes) {
                case 1: return write_result((uint8_t*) dst, src);
                case 2: return write_result((uint16_t*) dst, src);
                case 4: return write_result((uint32_t*) dst, src);
                case 8: return write_result((uint64_t*) dst, src);
            }
            break;
        case devices::TypeCode::Float:
            switch (info.bytes) {
                case 2: return write_result((half*) dst, src);
                case 4: return write_result((float*) dst, src);
                case 8: return write_result((double*) dst, src);
            }
            break;
        case devices::TypeCode::OpaqueHandle: break; // not supported
        case devices::TypeCode::BFloat:
            if (info.bytes == 2) {
                return write_result((bfloat16*) dst, src);
            }
            break;
        case devices::TypeCode::Complex:
            // TODO: implement complex conversions
            break;
        case devices::TypeCode::Bool: break; // not supported
        case devices::TypeCode::ArbitraryPrecisionInt: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::Rational:
            // later we might actually have a fixed precision rational.
        case devices::TypeCode::ArbitraryPrecisionRational:
            return write_result((rational_scalar_type*) dst, src);
        case devices::TypeCode::APRationalPolynomial:
            return write_result((rational_poly_scalar*) dst, src);
    }
    return false;
}

// For now, just cheat with half and bfloat16 and cast them to floats
static inline bool
convert_impl(void* dst, const TypeInfo& info, const half& src) noexcept
{
    return convert_impl(dst, info, float(src));
}

static inline bool
convert_impl(void* dst, const TypeInfo& info, const bfloat16& src) noexcept
{
    return convert_impl(dst, info, float(src));
}

bool scalars::dtl::scalar_convert_copy(
        void* dst,
        devices::TypeInfo dst_type,
        const Scalar& src
) noexcept
{
    auto src_info = src.type_info();
    switch (src_info.code) {
        case devices::TypeCode::Int:
            switch (src_info.bytes) {
                case 1:
                    return convert_impl(dst, dst_type, src.as_type<int8_t>());
                case 2:
                    return convert_impl(dst, dst_type, src.as_type<int16_t>());
                case 4:
                    return convert_impl(dst, dst_type, src.as_type<int32_t>());
                case 8:
                    return convert_impl(dst, dst_type, src.as_type<int64_t>());
            }
            break;
        case devices::TypeCode::UInt:
            switch (src_info.bytes) {
                case 1:
                    return convert_impl(dst, dst_type, src.as_type<uint8_t>());
                case 2:
                    return convert_impl(dst, dst_type, src.as_type<uint16_t>());
                case 4:
                    return convert_impl(dst, dst_type, src.as_type<uint32_t>());
                case 8:
                    return convert_impl(dst, dst_type, src.as_type<uint64_t>());
            }
            break;
        case devices::TypeCode::Float:
            switch (src_info.bytes) {
                case 2: return convert_impl(dst, dst_type, src.as_type<half>());
                case 4:
                    return convert_impl(dst, dst_type, src.as_type<float>());
                case 8:
                    return convert_impl(dst, dst_type, src.as_type<double>());
            }
            break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat:
            if (src_info.bytes == 2) {
                return convert_impl(dst, dst_type, src.as_type<bfloat16>());
            }
            break;
        case devices::TypeCode::Complex: break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::ArbitraryPrecision: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::Rational:
            // Later we might have a fixed precision rational.
        case devices::TypeCode::ArbitraryPrecisionRational:
            return convert_impl(
                    dst,
                    dst_type,
                    src.as_type<rational_scalar_type>()
            );
        case devices::TypeCode::Polynomial:
            return convert_impl(
                    dst,
                    dst_type,
                    src.as_type<rational_poly_scalar>()
            );
    }

    return false;
}



bool rpy::scalars::dtl::scalar_convert_copy(
        void* dst,
        devices::TypeInfo dst_type,
        const void* src,
        devices::TypeInfo src_type
) noexcept
{
   switch (src_type.code) {
        case devices::TypeCode::Int:
            switch (src_type.bytes) {
                case 1:
                    return convert_impl(dst, dst_type, *((int8_t*) src));
                case 2:
                    return convert_impl(dst, dst_type, *((int16_t*) src));
                case 4:
                    return convert_impl(dst, dst_type, *((int32_t*) src));
                case 8:
                    return convert_impl(dst, dst_type, *((int64_t*) src));
            }
            break;
        case devices::TypeCode::UInt:
            switch (src_type.bytes) {
                case 1:
                    return convert_impl(dst, dst_type, *((uint8_t*) src));
                case 2:
                    return convert_impl(dst, dst_type, *((uint16_t*) src));
                case 4:
                    return convert_impl(dst, dst_type, *((uint32_t*) src));
                case 8:
                    return convert_impl(dst, dst_type, *((uint64_t*) src));
            }
            break;
        case devices::TypeCode::Float:
            switch (src_type.bytes) {
                case 2: return convert_impl(dst, dst_type, *((half*) src));
                case 4:
                    return convert_impl(dst, dst_type, *((float*) src));
                case 8:
                    return convert_impl(dst, dst_type, *((double*) src));
            }
            break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat:
            if (src_type.bytes == 2) {
                return convert_impl(dst, dst_type, *((bfloat16*) src));
            }
            break;
        case devices::TypeCode::Complex: break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::ArbitraryPrecision: break;
        case devices::TypeCode::ArbitraryPrecisionUInt: break;
        case devices::TypeCode::ArbitraryPrecisionFloat: break;
        case devices::TypeCode::ArbitraryPrecisionComplex: break;
        case devices::TypeCode::Rational:
            // Later we might have a fixed precision rational.
        case devices::TypeCode::ArbitraryPrecisionRational:
            return convert_impl(
                    dst,
                    dst_type,
                    *((rational_scalar_type*) src)
            );
        case devices::TypeCode::Polynomial:
            return convert_impl(
                    dst,
                    dst_type,
                    *((rational_poly_scalar*) src)
            );
    }

    return false;
}
