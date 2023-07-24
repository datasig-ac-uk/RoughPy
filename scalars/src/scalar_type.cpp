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
// Created by user on 26/02/23.
//

#include <roughpy/scalars/scalar_type.h>

#include <mutex>
#include <ostream>
#include <unordered_map>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_pointer.h>

#include "RationalType.h"
#include "b_float_16_type.h"
#include "double_type.h"
#include "float_type.h"
#include "half_type.h"
#include "rational_poly_scalar_type.h"

using namespace rpy;
using namespace scalars;

const ScalarType* ScalarType::rational_type() const noexcept { return this; }

const ScalarType* ScalarType::host_type() const noexcept { return this; }

const ScalarType* ScalarType::for_id(const string& id)
{
    return ScalarType::of<double>();
}
const ScalarType* ScalarType::from_type_details(const BasicScalarInfo& details,
                                                const ScalarDeviceInfo& device)
{
    return ScalarType::of<double>();
}
Scalar ScalarType::from(long long int numerator,
                        long long int denominator) const
{
    if (denominator == 0) { RPY_THROW(std::invalid_argument, "division by zero"); }
    auto result = allocate(1);
    assign(result, numerator, denominator);
    return {result, flags::OwnedPointer};
}
void ScalarType::convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count,
                              const string& id) const
{
    if (!id.empty()) {
        try {
            const auto& conversion = get_conversion(id, this->id());
            conversion(out, in, count);
            return;
        } catch (std::runtime_error&) {}
    }

    convert_copy(out, in, 1);

    auto isize = itemsize();
    auto* out_p = out.raw_cast<char*>();
    for (dimn_t i = 1; i < count; ++i) {
        out_p += isize;
        convert_copy({this, out_p}, out, 1);
    }
}
Scalar ScalarType::parse(string_view str) const
{
    RPY_THROW(std::runtime_error, "could not parse string");
}
Scalar ScalarType::one() const { return from(1, 1); }
Scalar ScalarType::mone() const { return from(-1, 1); }
Scalar ScalarType::zero() const { return from(0, 1); }
Scalar ScalarType::copy(ScalarPointer source) const
{
    auto result = allocate(1);
    convert_copy(result, source, 1);

    return {result, flags::OwnedPointer};
}
Scalar ScalarType::add(ScalarPointer lhs, ScalarPointer rhs) const
{
    auto result = copy(lhs);
    add_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::sub(ScalarPointer lhs, ScalarPointer rhs) const
{
    auto result = copy(lhs);
    sub_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::mul(ScalarPointer lhs, ScalarPointer rhs) const
{
    auto result = copy(lhs);
    mul_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::div(ScalarPointer lhs, ScalarPointer rhs) const
{
    auto result = copy(lhs);
    div_inplace(result.to_mut_pointer(), rhs);
    return result;
}

bool ScalarType::is_zero(ScalarPointer arg) const
{
    return arg.is_null() || are_equal(arg, zero().to_pointer());
}
void ScalarType::print(ScalarPointer arg, std::ostream& os) const
{
    os << to_scalar_t(arg);
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<float>::get_type() noexcept
{
    static const FloatType ftype;
    return &ftype;
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<double>::get_type() noexcept
{
    static const DoubleType dtype;
    return &dtype;
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<rational_scalar_type>::get_type() noexcept
{
    static const RationalType rtype;
    return &rtype;
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<half>::get_type() noexcept
{
    static const HalfType htype;
    return &htype;
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<bfloat16>::get_type() noexcept
{
    static const BFloat16Type bf16type;
    return &bf16type;
}

const ScalarType*
rpy::scalars::dtl::scalar_type_holder<rational_poly_scalar>::get_type() noexcept
{
    static const RationalPolyScalarType rpolscaltype;
    return &rpolscaltype;
}

/*
 * All types should support conversion from the following,
 * but these are not represented as scalar types by themselves.
 * The last 3 are commented out, because they are to be implemented
 * separately.
 */

#define ADD_RES_PAIR(name, type, code)                                         \
    pair<string, ScalarTypeInfo>(                                              \
            (name),                                                            \
            {(name),                                                           \
             (name),                                                           \
             sizeof(type),                                                     \
             alignof(type),                                                    \
             BasicScalarInfo({(code), CHAR_BIT * sizeof(type), 1}),            \
             {ScalarDeviceType::CPU, 0}})

static const pair<string, ScalarTypeInfo> reserved[]
        = {ADD_RES_PAIR("i32", int32_t, ScalarTypeCode::Int),
           ADD_RES_PAIR("u32", uint32_t, ScalarTypeCode::UInt),
           ADD_RES_PAIR("i64", int64_t, ScalarTypeCode::Int),
           ADD_RES_PAIR("u64", uint64_t, ScalarTypeCode::UInt),
           ADD_RES_PAIR("isize", idimn_t, ScalarTypeCode::Int),
           ADD_RES_PAIR("usize", dimn_t, ScalarTypeCode::UInt),
           ADD_RES_PAIR("i16", int16_t, ScalarTypeCode::Int),
           ADD_RES_PAIR("u16", uint16_t, ScalarTypeCode::UInt),
           ADD_RES_PAIR("i8", int8_t, ScalarTypeCode::Int),
           ADD_RES_PAIR("u8", uint8_t, ScalarTypeCode::UInt)};

#undef ADD_RES_PAIR

#define I8IDX 8
#define U8IDX 9
#define I16IDX 6
#define U16IDX 7
#define I32IDX 0
#define U32IDX 1
#define I64IDX 2
#define U64IDX 3

static std::mutex s_scalar_type_cache_lock;
static std::unordered_map<string, const ScalarType*> s_scalar_type_cache{
        {string("f32"), ScalarType::of<float>()},
        {string("f64"), ScalarType::of<double>()},
        {string("f16"), ScalarType::of<half>()},
        {string("bf16"), ScalarType::of<bfloat16>()},
        {string("rational"), ScalarType::of<rational_scalar_type>()}
};

void rpy::scalars::register_type(const ScalarType* type)
{
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);

    const auto& identifier = type->id();
    for (const auto& i : reserved) {
        if (identifier == i.first) {
            RPY_THROW(std::runtime_error,"cannot register identifier " + identifier
                                     + ", it is reserved");
        }
    }

    auto& entry = s_scalar_type_cache[identifier];
    if (entry != nullptr) {
        RPY_THROW(std::runtime_error,"type with id " + identifier
                                 + " is already registered");
    }

    entry = type;
}
const ScalarType* rpy::scalars::get_type(const string& id)
{
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);
    auto found = s_scalar_type_cache.find(id);
    if (found != s_scalar_type_cache.end()) { return found->second; }

    for (const auto& rsv : reserved) {
        if (id == rsv.first) { return nullptr; }
    }

    RPY_THROW(std::runtime_error, "no type with id " + id);
}

const ScalarType* scalars::get_type(const string& id,
                                    const ScalarDeviceInfo& device)
{
    // TODO: Needs implementation
    return nullptr;
}

const ScalarTypeInfo& rpy::scalars::get_scalar_info(string_view id)
{

    for (const auto& rsv : reserved) {
        if (rsv.first == id) { return rsv.second; }
    }

    string ids(id);
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);
    auto found = s_scalar_type_cache.find(ids);
    if (found != s_scalar_type_cache.end()) { return found->second->info(); }

    RPY_THROW(std::runtime_error, "Unrecognised type id " + ids);
}

std::vector<const ScalarType*> rpy::scalars::list_types()
{
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);
    std::vector<const ScalarType*> result;
    result.reserve(s_scalar_type_cache.size());
    for (const auto& [id, type] : s_scalar_type_cache) {
        result.push_back(type);
    }
    return result;
}

#define ROUGHPY_MAKE_TYPE_ID_OF(TYPE, NAME)                                    \
    const string& rpy::scalars::dtl::type_id_of_impl<TYPE>::get_id() noexcept  \
    {                                                                          \
        static const string type_id(NAME);                                     \
        return type_id;                                                        \
    }

ROUGHPY_MAKE_TYPE_ID_OF(char, "i8")
ROUGHPY_MAKE_TYPE_ID_OF(unsigned char, "u8")
ROUGHPY_MAKE_TYPE_ID_OF(short, "i16")
ROUGHPY_MAKE_TYPE_ID_OF(unsigned short, "u16")
ROUGHPY_MAKE_TYPE_ID_OF(int, "i32")
ROUGHPY_MAKE_TYPE_ID_OF(unsigned int, "u32")
ROUGHPY_MAKE_TYPE_ID_OF(long long, "i64")
ROUGHPY_MAKE_TYPE_ID_OF(unsigned long long, "u64")
ROUGHPY_MAKE_TYPE_ID_OF(signed_size_type_marker, "isize")
ROUGHPY_MAKE_TYPE_ID_OF(unsigned_size_type_marker, "usize")

ROUGHPY_MAKE_TYPE_ID_OF(half, "f16")
ROUGHPY_MAKE_TYPE_ID_OF(bfloat16, "bf16")
ROUGHPY_MAKE_TYPE_ID_OF(float, "f32")
ROUGHPY_MAKE_TYPE_ID_OF(double, "f64")
ROUGHPY_MAKE_TYPE_ID_OF(rational_scalar_type, "Rational")
ROUGHPY_MAKE_TYPE_ID_OF(rational_poly_scalar, "RationalPoly")

#undef ROUGHPY_MAKE_TYPE_ID_OF

#define MAKE_CONVERSION_FUNCTION(SRC, DST, SRC_T, DST_T)                       \
    static void SRC##_to_##DST(ScalarPointer dst, ScalarPointer src,           \
                               dimn_t count)                                   \
    {                                                                          \
        const auto* src_p = src.raw_cast<const SRC_T>();                       \
        auto* dst_p = dst.raw_cast<DST_T>();                                   \
                                                                               \
        for (dimn_t i = 0; i < count; ++i) {                                   \
            ::new (dst_p++) DST_T(src_p[i]);                                   \
        }                                                                      \
    }

MAKE_CONVERSION_FUNCTION(f32, f64, float, double)
MAKE_CONVERSION_FUNCTION(f64, f32, double, float)
MAKE_CONVERSION_FUNCTION(i32, f32, int, float)
MAKE_CONVERSION_FUNCTION(i32, f64, int, double)
MAKE_CONVERSION_FUNCTION(i64, f32, long long, float)
MAKE_CONVERSION_FUNCTION(i64, f64, long long, double)
MAKE_CONVERSION_FUNCTION(i16, f32, short, float)
MAKE_CONVERSION_FUNCTION(i16, f64, short, double)
MAKE_CONVERSION_FUNCTION(i8, f32, char, float)
MAKE_CONVERSION_FUNCTION(i8, f64, char, double)
MAKE_CONVERSION_FUNCTION(isize, f32, idimn_t, float)
MAKE_CONVERSION_FUNCTION(isize, f64, idimn_t, double)

#undef MAKE_CONVERSION_FUNCTION

using pair_type = std::pair<string, conversion_function>;

#define ADD_DEF_CONV(SRC, DST)                                                 \
    pair_type { string(#SRC "->" #DST), conversion_function(&SRC##_to_##DST) }

static std::mutex s_conversion_lock;
static std::unordered_map<string, conversion_function> s_conversion_cache{
        ADD_DEF_CONV(f32, f64),   ADD_DEF_CONV(f64, f32),
        ADD_DEF_CONV(i32, f32),   ADD_DEF_CONV(i32, f64),
        ADD_DEF_CONV(i64, f32),   ADD_DEF_CONV(i64, f64),
        ADD_DEF_CONV(i16, f32),   ADD_DEF_CONV(i16, f64),
        ADD_DEF_CONV(i8, f32),    ADD_DEF_CONV(i8, f64),
        ADD_DEF_CONV(isize, f32), ADD_DEF_CONV(isize, f64)};

#undef ADD_DEF_CONV

static inline string type_ids_to_key(const string& src_type,
                                     const string& dst_type)
{
    return src_type + "->" + dst_type;
}

const conversion_function& rpy::scalars::get_conversion(const string& src_id,
                                                        const string& dst_id)
{
    std::lock_guard<std::mutex> access(s_conversion_lock);

    auto found = s_conversion_cache.find(type_ids_to_key(src_id, dst_id));
    if (found != s_conversion_cache.end()) { return found->second; }

    RPY_THROW(std::runtime_error,"no conversion function from " + src_id + " to "
                             + dst_id);
}
void rpy::scalars::register_conversion(const string& src_id,
                                       const string& dst_id,
                                       conversion_function converter)
{
    std::lock_guard<std::mutex> access(s_conversion_lock);

    auto& found = s_conversion_cache[type_ids_to_key(src_id, dst_id)];
    if (found != nullptr) {
        RPY_THROW(std::runtime_error,"conversion from " + src_id + " to " + dst_id
                                 + " already registered");
    } else {
        found = std::move(converter);
    }
}

std::unique_ptr<RandomGenerator>
ScalarType::get_rng(const string& bit_generator, Slice<uint64_t> seed) const
{
    RPY_THROW(std::runtime_error,
            "no random number generators are defined for this scalar type");
}

std::unique_ptr<BlasInterface> ScalarType::get_blas() const
{
    RPY_THROW(std::runtime_error, "No blas/lapack available for this scalar type");
}

const std::string& rpy::scalars::id_from_basic_info(const BasicScalarInfo& info)
{
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);

    // Lanes are ignored in determining type.

    switch (info.code) {
        case ScalarTypeCode::Int:
            switch (info.bits) {
                case 8: return reserved[I8IDX].second.id;
                case 16: return reserved[I16IDX].second.id;
                case 32: return reserved[I32IDX].second.id;
                case 64: return reserved[I64IDX].second.id;
                default: RPY_THROW(std::runtime_error, "unsupported integer type");
            }
        case ScalarTypeCode::UInt:
            switch (info.bits) {
                case 8: return reserved[U8IDX].second.id;
                case 16: return reserved[U16IDX].second.id;
                case 32: return reserved[U32IDX].second.id;
                case 64: return reserved[U64IDX].second.id;
                default: RPY_THROW(std::runtime_error, "unsupported integer type");
            }
        case ScalarTypeCode::Float:
            switch (info.bits) {
                case 32: return ScalarType::of<float>()->id();
                case 64: return ScalarType::of<double>()->id();
                default: RPY_THROW(std::runtime_error, "unsupported float type");
            }
        default: RPY_THROW(std::runtime_error, "unsupported scalar type");
    }

    RPY_UNREACHABLE();
}

#undef I8IDX
#undef I16IDX
#undef I32IDX
#undef I64IDX
#undef U8IDX
#undef U16IDX
#undef U32IDX
#undef U64IDX
