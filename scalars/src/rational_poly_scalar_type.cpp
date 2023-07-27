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
// Created by user on 23/05/23.
//

#include "rational_poly_scalar_type.h"
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type_helper.h>

#include <roughpy/core/alloc.h>

static inline rpy::scalars::rational_poly_scalar try_convert(
        rpy::scalars::ScalarPointer arg,
        const rpy::scalars::ScalarType* type = nullptr
)
{
    return ::rpy::scalars::try_convert<rpy::scalars::rational_poly_scalar>(
            arg, type
    );
}

namespace rpy {
namespace scalars {

const ScalarType* RationalPolyScalarType::rational_type() const noexcept
{
    return ScalarType::of<rational_scalar_type>();
}
const ScalarType* RationalPolyScalarType::host_type() const noexcept
{
    return this;
}

Scalar RationalPolyScalarType::from(
        long long int numerator, long long int denominator
) const
{
    return Scalar(
            this,
            rational_poly_scalar(
                    rational_scalar_type(numerator)
                    / rational_scalar_type(denominator)
            )
    );
}
ScalarPointer RationalPolyScalarType::allocate(std::size_t count) const
{
    if (count == 1) {
        return ScalarPointer(
                this, new rational_poly_scalar(),
                flags::IsMutable | flags::OwnedPointer
        );
    } else {
        return ScalarPointer(
                this, new rational_poly_scalar[count](),
                flags::IsMutable | flags::OwnedPointer
        );
    }
}
void RationalPolyScalarType::free(ScalarPointer pointer, std::size_t count)
        const
{
    if (!pointer.is_null()) {
        if (count == 1) {
            delete pointer.template raw_cast<rational_poly_scalar>();
        } else {
            delete[] pointer.template raw_cast<rational_poly_scalar>();
        }
    }
}
void RationalPolyScalarType::swap(ScalarPointer lhs, ScalarPointer rhs) const
{
    RPY_CHECK(!(lhs.is_null() ^ rhs.is_null()));
    RPY_CHECK(
            (lhs.type() == nullptr || rhs.type() == nullptr)
            || lhs.type() == rhs.type()
    );
    RPY_CHECK(lhs.type() == this);
    RPY_CHECK(!lhs.is_const() && !rhs.is_const());

    std::swap(
            *lhs.raw_cast<rational_poly_scalar*>(),
            *rhs.raw_cast<rational_poly_scalar*>()
    );
}
void RationalPolyScalarType::convert_copy(
        ScalarPointer dst, ScalarPointer src, dimn_t count
) const
{
    if (src.type() == nullptr) {
        RPY_THROW(std::invalid_argument, "source type cannot be null");
    }
    convert_copy(dst, src.ptr(), count, src.type()->id());
}

template <typename F>
static inline void
convert_copy_ext(ScalarPointer& out, const void* in, dimn_t count)
{
    const auto* iptr = static_cast<const F*>(in);
    auto* optr = out.template raw_cast<rational_poly_scalar*>();

    for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
        construct_inplace<rational_poly_scalar>(optr, *iptr);
    }
}

void RationalPolyScalarType::convert_copy(
        void* out, const void* in, std::size_t count, BasicScalarInfo info
) const
{
    ScalarPointer optr(this, out);
    switch (info.code) {
        case ScalarTypeCode::Int:
            switch (info.bits) {
                case 8:
                    convert_copy_ext<int8_t>(optr, in, info.lanes * count);
                    break;
                case 16:
                    convert_copy_ext<int16_t>(optr, in, info.lanes * count);
                    break;
                case 32:
                    convert_copy_ext<int32_t>(optr, in, info.lanes * count);
                    break;
                case 64:
                    convert_copy_ext<int64_t>(optr, in, info.lanes * count);
                    break;
                case 128:
                default:
                    RPY_THROW(
                            std::runtime_error,
                            "invalid bit configuration for integer type"
                    );
            }
            break;
        case ScalarTypeCode::UInt:
            switch (info.bits) {
                case 8:
                    convert_copy_ext<uint8_t>(optr, in, info.lanes * count);
                    break;
                case 16:
                    convert_copy_ext<uint16_t>(optr, in, info.lanes * count);
                    break;
                case 32:
                    convert_copy_ext<uint32_t>(optr, in, info.lanes * count);
                    break;
                case 64:
                    convert_copy_ext<uint64_t>(optr, in, info.lanes * count);
                    break;
                case 128:
                default:
                    RPY_THROW(
                            std::runtime_error,
                            "invalid bit configuration for integer type"
                    );
            }
            break;
        case ScalarTypeCode::Float:
            switch (info.bits) {
                    //                case 16:
                    //                    convert_copy_ext<half>(optr, in,
                    //                    info.lanes * count); break;
                case 32:
                    convert_copy_ext<float>(optr, in, info.lanes * count);
                    break;
                case 64:
                    convert_copy_ext<double>(optr, in, info.lanes * count);
                    break;
                default:
                    RPY_THROW(
                            std::runtime_error,
                            "invalid bit configuration for float type"
                    );
            }
            break;
        case ScalarTypeCode::BFloat:
            switch (info.bits) {
                    //                case 16:
                    //                    convert_copy_ext<bfloat16>(optr, in,
                    //                    info.lanes * count); break;
                default:
                    RPY_THROW(
                            std::runtime_error,
                            "invalid bit configuration for bfloat type"
                    );
            }
            break;
        case ScalarTypeCode::Bool:
        case ScalarTypeCode::OpaqueHandle: break;
        case ScalarTypeCode::Complex:
        default: RPY_THROW(std::runtime_error, "unsupported scalar type");
    }
}
void RationalPolyScalarType::convert_copy(
        void* out, ScalarPointer in, std::size_t count
) const
{
    RPY_DBG_ASSERT(out != nullptr);
    RPY_DBG_ASSERT(!in.is_null());
    const auto* type = in.type();
    RPY_CHECK(type != nullptr);

    if (type == this) {
        const auto* in_begin = in.template raw_cast<const scalar_type>();
        const auto* in_end = in_begin + count;
        std::copy(in_begin, in_end, static_cast<scalar_type*>(out));
    } else {
        auto cv = get_conversion(type->id(), this->id());
        ScalarPointer out_ptr{this, out};
        cv(out_ptr, in, count);
    }
}
void RationalPolyScalarType::convert_copy(
        ScalarPointer out, const void* in, std::size_t count, const string& id
) const
{
    if (id == this->id()) {
        const auto* in_begin = reinterpret_cast<const scalar_type*>(in);
        const auto* in_end = in_begin + count;
        std::copy(in_begin, in_end, out.template raw_cast<scalar_type*>());
        return;
    }

    if (id == "f64") {
        return convert_copy_ext<double>(out, in, count);
    } else if (id == "f32") {
        return convert_copy_ext<float>(out, in, count);
    } else if (id == "i32") {
        return convert_copy_ext<int>(out, in, count);
    } else if (id == "u32") {
        return convert_copy_ext<unsigned int>(out, in, count);
    } else if (id == "i64") {
        return convert_copy_ext<long long>(out, in, count);
    } else if (id == "u64") {
        return convert_copy_ext<unsigned long long>(out, in, count);
    } else if (id == "isize") {
        return convert_copy_ext<std::ptrdiff_t>(out, in, count);
    } else if (id == "usize") {
        return convert_copy_ext<std::size_t>(out, in, count);
    } else if (id == "i16") {
        return convert_copy_ext<short>(out, in, count);
    } else if (id == "u16") {
        return convert_copy_ext<unsigned short>(out, in, count);
    } else if (id == "i8") {
        return convert_copy_ext<char>(out, in, count);
    } else if (id == "u8") {
        return convert_copy_ext<unsigned char>(out, in, count);
    }

    // If we're here, then it is a non-standard type
    const auto& conversion = get_conversion(id, this->id());
    conversion(out, {nullptr, in}, count);
}
void RationalPolyScalarType::convert_fill(
        ScalarPointer out, ScalarPointer in, dimn_t count, const string& id
) const
{
    ScalarType::convert_fill(out, in, count, id);
}
Scalar RationalPolyScalarType::parse(string_view str) const
{
    return ScalarType::parse(str);
}
Scalar RationalPolyScalarType::one() const
{
    return Scalar(this, scalar_type(1));
}
Scalar RationalPolyScalarType::mone() const
{
    return Scalar(this, scalar_type(-1));
}
Scalar RationalPolyScalarType::zero() const
{
    return Scalar(this, scalar_type(0));
}
scalar_t RationalPolyScalarType::to_scalar_t(ScalarPointer arg) const
{
    return 0;
}
void RationalPolyScalarType::assign(
        ScalarPointer target, long long int numerator, long long int denominator
) const
{
    construct_inplace(
            target.template raw_cast<scalar_type*>(),
            rational_scalar_type(numerator) / rational_scalar_type(denominator)
    );
}
Scalar RationalPolyScalarType::copy(ScalarPointer source) const
{
    return Scalar(this, ::try_convert(source));
}
Scalar RationalPolyScalarType::uminus(ScalarPointer arg) const
{
    return Scalar(this, -::try_convert(arg));
}
Scalar RationalPolyScalarType::add(ScalarPointer lhs, ScalarPointer rhs) const
{
    RPY_DBG_ASSERT(lhs.type() == this);
    if (!lhs) { return copy(rhs); }
    return Scalar(
            this, *lhs.raw_cast<const scalar_type*>() + ::try_convert(rhs)
    );
}
Scalar RationalPolyScalarType::sub(ScalarPointer lhs, ScalarPointer rhs) const
{
    if (!lhs) { return uminus(rhs); }
    return Scalar(
            this, *lhs.raw_cast<const scalar_type*>() - ::try_convert(rhs)
    );
}
Scalar RationalPolyScalarType::mul(ScalarPointer lhs, ScalarPointer rhs) const
{
    if (!lhs) { return zero(); }
    return Scalar(
            this,
            *lhs.template raw_cast<const scalar_type*>() * ::try_convert(rhs)
    );
}
Scalar RationalPolyScalarType::div(ScalarPointer lhs, ScalarPointer rhs) const
{
    if (!lhs) { return zero(); }
    if (rhs.is_null()) { RPY_THROW(std::runtime_error, "division by zero"); }

    auto divisor = try_convert<rational_scalar_type>(rhs);

    if (divisor == rational_scalar_type(0)) {
        RPY_THROW(std::runtime_error, "division by zero");
    }

    return Scalar(this, *lhs.template raw_cast<scalar_type>() / divisor);
}
void RationalPolyScalarType::add_inplace(ScalarPointer lhs, ScalarPointer rhs)
        const
{
    RPY_CHECK(lhs);
    auto* ptr = lhs.raw_cast<scalar_type*>();
    *ptr += ::try_convert(rhs);
}
void RationalPolyScalarType::sub_inplace(ScalarPointer lhs, ScalarPointer rhs)
        const
{
    RPY_CHECK(lhs);
    auto* ptr = lhs.raw_cast<scalar_type*>();
    *ptr -= ::try_convert(rhs);
}
void RationalPolyScalarType::mul_inplace(ScalarPointer lhs, ScalarPointer rhs)
        const
{
    RPY_CHECK(lhs);
    auto* ptr = lhs.raw_cast<scalar_type*>();
    *ptr *= ::try_convert(rhs);
}
void RationalPolyScalarType::div_inplace(ScalarPointer lhs, ScalarPointer rhs)
        const
{
    RPY_CHECK(lhs);
    auto* ptr = lhs.raw_cast<scalar_type*>();

    if (rhs.is_null()) { RPY_THROW(std::runtime_error, "division by zero"); }

    auto divisor = try_convert<rational_scalar_type>(rhs);

    if (divisor == rational_scalar_type(0)) {
        RPY_THROW(std::runtime_error, "division by zero");
    }

    *ptr /= divisor;
}
bool RationalPolyScalarType::is_zero(ScalarPointer arg) const
{
    return !static_cast<bool>(arg)
            || *arg.raw_cast<const scalar_type*>() == scalar_type(0);
}
bool RationalPolyScalarType::are_equal(ScalarPointer lhs, ScalarPointer rhs)
        const noexcept
{
    return *lhs.raw_cast<const scalar_type*>() == ::try_convert(rhs);
}
void RationalPolyScalarType::print(ScalarPointer arg, std::ostream& os) const
{
    if (!arg) {
        os << "{ }";
    } else {
        os << *arg.raw_cast<const scalar_type*>();
    }
}
std::unique_ptr<RandomGenerator> RationalPolyScalarType::get_rng(
        const string& bit_generator, Slice<uint64_t> seed
) const
{
    RPY_THROW(std::runtime_error, "no rng for rational polynomial scalars");
}
std::unique_ptr<BlasInterface> RationalPolyScalarType::get_blas() const
{
    RPY_THROW(
            std::runtime_error,
            "no blas implementation for rational polynomial scalars"
    );
}
std::vector<byte> RationalPolyScalarType::to_raw_bytes(
        const ScalarPointer& ptr, dimn_t count
) const
{
    RPY_THROW(std::runtime_error, "not implemented");
}
ScalarPointer RationalPolyScalarType::from_raw_bytes(
        Slice<byte> raw_bytes, dimn_t count
) const
{
    RPY_THROW(std::runtime_error, "not implemented");
}
void RationalPolyScalarType::add_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<rational_poly_scalar>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l + r; }
    );
}
void RationalPolyScalarType::sub_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<rational_poly_scalar>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l - r; }
    );
}
void RationalPolyScalarType::mul_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<rational_poly_scalar>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l * r; }
    );
}
void RationalPolyScalarType::div_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
    //    impl_helpers::binary_into_buffer<rational_poly_scalar>(
    //            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l / r;
    //            }
    //    );
}
}// namespace scalars
}// namespace rpy
