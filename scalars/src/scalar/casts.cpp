

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

#include <roughpy/scalars/scalar.h>

#include "casts.h"
#include "do_macro.h"
#include <roughpy/core/alloc.h>

#include "builtin_scalars.h"

using namespace rpy;
using namespace scalars;
using devices::TypeInfo;

template <typename D, typename T>
static inline enable_if_t<
        is_constructible_v<D, const T&>
                && is_trivially_default_constructible_v<D>,
        bool>
write_result(D* dst, const T* src, dimn_t count) noexcept
{
    for (dimn_t i = 0; i < count; ++i) { construct_inplace(dst++, *(src++)); }
    return true;
}

template <typename D, typename T>
static inline void assign_result(D& result, const T& arg)
{
    result = static_cast<D>(arg);
}

template <typename D>
static inline void assign_result(D& result, const Half& arg)
{
    result = static_cast<D>(static_cast<float>(arg));
}

template <typename D>
static inline void assign_result(D& result, const BFloat16& arg)
{
    result = static_cast<D>(static_cast<float>(arg));
}

template <typename D, typename T>
static inline enable_if_t<
        is_constructible_v<D, const T&>
                && !is_trivially_default_constructible_v<D>,
        bool>
write_result(D* dst, const T* src, dimn_t count) noexcept
{
    try {
        for (dimn_t i = 0; i < count; ++i) { assign_result(dst[i], src[i]); }
    } catch (...) {
        return false;
    }
    return true;
}

template <typename D, typename T>
static inline enable_if_t<!is_constructible_v<D, const T&>, bool>
write_result(D*, const T*, dimn_t) noexcept
{
    return false;
}

static inline bool write_single_poly(APPolyRat* dst, const APPolyRat& value)
{
    *dst = value;
    return true;
}

template <typename D>
static inline bool write_single_poly(D* dst, const APPolyRat& value)
{
    if (value.empty()) {
        auto tmp = 0;
        write_result(dst, &tmp, 1);
        return true;
    }

    if (value.size() == 1) {
        auto kv = value.begin();
        if (kv->first == Monomial()) {
            // Try converting via rational
            return write_result(dst, &kv->second, 1);
        }
    }

    return false;
}

template <typename D>
static inline bool
write_result(D* dst, const APPolyRat* value, dimn_t count) noexcept
{
    try {
        for (dimn_t i = 0; i < count; ++i) {
            if (!write_single_poly(dst++, value[i])) { return false; }
        }
    } catch (...) {
        return false;
    }

    return true;
}

template <typename T>
static inline bool convert_impl(
        void* dst,
        const PackedScalarType& info,
        const T* src,
        dimn_t count
) noexcept
{
#define X(TP) return write_result((TP*) dst, src, count)
    DO_FOR_EACH_X(type_info_from(info))
#undef X
    return false;
}
//
//// For now, just cheat with Half and BFloat16 and cast them to floats
// static inline bool
// convert_impl(void* dst, const TypeInfo& info, const Half* src, dimn_t count)
//         noexcept
//{
//     return convert_impl(dst, info, src, count);
// }
//
// static inline bool
// convert_impl(void* dst, const TypeInfo& info, const BFloat16* src, dimn_t
//                                                                            count)
//                                                                            noexcept
//{
//     return convert_impl(dst, info, src, count);
// }

bool scalars::dtl::scalar_convert_copy(
        void* dst,
        PackedScalarType dst_type,
        const Scalar& src
) noexcept
{
    //     auto src_info = src.type_info();
    // #define X(TP) return convert_impl(dst, dst_type, (const TP*)
    // src.pointer(), 1)
    //     DO_FOR_EACH_X(src_info)
    // #undef X
    return false;
}

bool rpy::scalars::dtl::scalar_convert_copy(
        void* dst,
        PackedScalarType dst_type,
        const void* src,
        PackedScalarType src_type,
        dimn_t count
) noexcept
{
    const auto tp = type_info_from(src_type);
#define X(TP) return convert_impl<TP>(dst, dst_type, (const TP*) src, count)
    DO_FOR_EACH_X(tp)
#undef X
    return false;
}

namespace {

template <typename T>
constexpr enable_if_t<
        is_trivially_constructible_v<T> && is_standard_layout_v<T>,
        bool>
assign_rational(T* dst, int64_t num, int64_t denom) noexcept
{
    construct_inplace(dst, T(num) / T(denom));
    return true;
}

inline bool assign_rational(
        ArbitraryPrecisionRational* dst,
        int64_t num,
        int64_t denom
) noexcept
{
    construct_inplace(dst, num, denom);
    return true;
}
inline bool assign_rational(APPolyRat* dst, int64_t num, int64_t denom) noexcept
{
    construct_inplace(dst, ArbitraryPrecisionRational(num, denom));
    return true;
}

constexpr bool assign_rational(Half* dst, int64_t num, int64_t denom) noexcept
{
    construct_inplace(dst, static_cast<float>(num) / denom);
    return true;
}

constexpr bool
assign_rational(BFloat16* dst, int64_t num, int64_t denom) noexcept
{
    construct_inplace(dst, static_cast<float>(num) / denom);
    return true;
}

}// namespace

bool scalars::dtl::scalar_assign_rational(
        void* dst,
        devices::TypeInfo dst_type,
        int64_t numerator,
        int64_t denominator
)
{
#define X(TP) return assign_rational((TP*) dst, numerator, denominator)
    DO_FOR_EACH_X(dst_type)
#undef X
    return false;
}
