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
// Created by sam on 13/03/23.
//

#include "RationalType.h"

#include <roughpy/device/device_handle.h>

#include <algorithm>
#include <cmath>
#include <ostream>
#include <utility>

#include "scalar_type_helper.h"

#if RPY_USING_GMP
#  include <gmp.h>
#else

#  include <iterator>

#endif

#include <boost/endian.hpp>

using namespace rpy;
using namespace rpy::scalars;

std::unique_ptr<RandomGenerator> RationalType::get_mt19937_generator(
        const ScalarType* type, Slice<uint64_t> seed
)
{
    return nullptr;
}
std::unique_ptr<RandomGenerator>
RationalType::get_pcg_generator(const ScalarType* type, Slice<uint64_t> seed)
{
    return std::unique_ptr<RandomGenerator>();
}

RationalType::RationalType()
    : helper({
            "Rational",
            "rational",
            sizeof(rational_scalar_type),
            alignof(rational_scalar_type),
            {
                                    ScalarTypeCode::ArbitraryPrecisionRational,
                                    0, 0,
                                    },
            {       devices::DeviceType::CPU, 0   }
},
            devices::get_host_device())
{}
ScalarPointer RationalType::allocate(std::size_t count) const
{
    if (count == 1) {
        return ScalarPointer(
                this, new rational_scalar_type,
                flags::IsMutable | flags::OwnedPointer
        );
    } else {
        return ScalarPointer(
                this, new rational_scalar_type[count],
                flags::IsMutable | flags::OwnedPointer
        );
    }
}
void RationalType::free(ScalarPointer pointer, std::size_t count) const
{
    if (!pointer.is_null()) {
        if (count == 1) {
            delete pointer.template raw_cast<rational_scalar_type>();
        } else {
            delete[] pointer.template raw_cast<rational_scalar_type>();
        }
    }
}

RationalType::scalar_type RationalType::try_convert(ScalarPointer other) const
{
    if (other.is_null()) { return scalar_type(0); }
    if (other.type() == this) {
        return *other.template raw_cast<const scalar_type>();
    }

    const ScalarType* type = other.type();
    if (type == nullptr) {
        RPY_THROW(std::runtime_error, "null type for non-zero value");
    }

    auto cv = get_conversion(type->id(), this->id());
    if (cv) {
        scalar_type result;
        ScalarPointer result_ptr{this, &result};
        cv(result_ptr, other, 1);
        return result;
    }

    RPY_THROW(
            std::runtime_error,
            "could not convert " + type->info().name + " to scalar type "
                    + info().name
    );
}

void RationalType::convert_copy(
        ScalarPointer dst, ScalarPointer src, dimn_t count
) const
{
    helper::copy_convert(dst, src, count);
}

template <typename F>
static inline void
convert_copy_ext(ScalarPointer& out, const void* in, std::size_t count)
{
    const auto* iptr = static_cast<const F*>(in);
    auto* optr = static_cast<rational_scalar_type*>(out.ptr());

    for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
        ::new (optr) rational_scalar_type(static_cast<float>(*iptr));
    }
}

scalar_t RationalType::to_scalar_t(ScalarPointer arg) const
{
    return static_cast<scalar_t>(*arg.raw_cast<const scalar_type*>());
}
void RationalType::assign(
        ScalarPointer target, long long int numerator, long long int denominator
) const
{
    *target.raw_cast<scalar_type*>() = scalar_type(numerator) / denominator;
}


bool RationalType::are_equal(ScalarPointer lhs, ScalarPointer rhs)
        const noexcept
{
    return *lhs.raw_cast<const scalar_type*>() == helper::try_convert(rhs);
}
Scalar
RationalType::from(long long int numerator, long long int denominator) const
{
    return Scalar(this, scalar_type(numerator) / denominator);
}
void RationalType::convert_fill(
        ScalarPointer out, ScalarPointer in, dimn_t count, const string& id
) const
{
    ScalarType::convert_fill(out, in, count, id);
}
Scalar RationalType::one() const { return Scalar(this, scalar_type(1)); }
Scalar RationalType::mone() const { return Scalar(this, scalar_type(-1)); }
Scalar RationalType::zero() const { return Scalar(this, scalar_type(0)); }


bool RationalType::is_zero(ScalarPointer arg) const
{
    return !static_cast<bool>(arg)
            || *arg.raw_cast<const scalar_type*>() == scalar_type(0);
}
void RationalType::print(ScalarPointer arg, std::ostream& os) const
{
    if (!arg) {
        os << 0.0;
    } else {
        os << *arg.raw_cast<const scalar_type*>();
    }
}
std::unique_ptr<RandomGenerator>
RationalType::get_rng(const string& bit_generator, Slice<uint64_t> seed) const
{
    return ScalarType::get_rng(bit_generator, seed);
}
void RationalType::swap(ScalarPointer lhs, ScalarPointer rhs, dimn_t count)
        const
{

    if (lhs.is_null() ^ rhs.is_null()) {
        RPY_THROW(std::runtime_error, "one of the pointers is null");
    }

    if (lhs.type() != rhs.type()) {
        RPY_THROW(std::runtime_error, "cannot swap scalars of different types");
    }

    if (lhs.type() != this && lhs.type() != nullptr) {
        return lhs.type()->swap(lhs, rhs, 0);
    }

    if (lhs.is_const() || rhs.is_const()) {
        RPY_THROW(std::runtime_error, "one or both of the scalars is const");
    }

    auto* lptr = lhs.raw_cast<scalar_type*>();
    auto* rptr = rhs.raw_cast<scalar_type*>();

    for (dimn_t i=0; i<count; ++i) {
        std::swap(lptr[i], rptr[i]);
    }
}
std::vector<byte>
RationalType::to_raw_bytes(const ScalarPointer& ptr, dimn_t count) const
{
    const auto* raw = ptr.raw_cast<const rational_scalar_type*>();
    std::vector<byte> result;
    result.reserve(
            count * sizeof(rational_scalar_type)
    );// Should be a reasonable
      // approximation of the final size

    auto push_items = [&result](auto item) {
#if RPY_USING_GMP
        auto size = static_cast<int64_t>(item->_mp_size) * sizeof(mp_limb_t);
        auto n_bytes = (size >= 0) ? size : -size;// abs(size) is ambiguous?
#else
        auto size = static_cast<int64_t>(item.backend().size());
        auto n_bytes = size * sizeof(boost::multiprecision::limb_type);
        size = item.sign() ? -n_bytes : n_bytes;
#endif
        const auto* sz_ptr = reinterpret_cast<const byte*>(&size);
        for (dimn_t i = 0; i < sizeof(int64_t); ++i) {
            result.push_back(sz_ptr[i]);
        }

#if RPY_USING_GMP
        auto curr_size = result.size();
        result.resize(curr_size + n_bytes);
        auto* write = result.data() + curr_size;
        size_t actually_written;
        mpz_export(write, &actually_written, -1, sizeof(byte), -1, 0, item);
        RPY_DBG_ASSERT(actually_written == n_bytes);
#else
        export_bits(item, std::back_inserter(result), CHAR_BIT, false);
#endif
    };

    for (dimn_t i = 0; i < count; ++i) {
#if RPY_USING_GMP
        const auto& item = raw[i].backend();
        push_items(mpq_numref(item.data()));
        push_items(mpq_denref(item.data()));
#else
        const auto& item = raw[i];
        push_items(numerator(item));
        push_items(denominator(item));
#endif
    }
    return result;
}

ScalarPointer
RationalType::from_raw_bytes(Slice<byte> raw_bytes, dimn_t count) const
{
    // TODO: These implementations are probably not completely correct

    ScalarPointer out = allocate(count);
    auto* raw = out.raw_cast<rational_scalar_type*>();

    dimn_t pos = 0;
    const auto limit = raw_bytes.size();

    auto unpack_limb = [&pos, &limit](auto& dst, const byte* src) {
        RPY_CHECK(pos + sizeof(int64_t) <= limit);

        int64_t size;
        std::memcpy(&size, src, sizeof(int64_t));
        pos += sizeof(int64_t);
        if (size == 0) { return; }

        auto n_bytes = abs(size);
        src += sizeof(int64_t);
        RPY_CHECK(pos + n_bytes <= limit);
#if RPY_USING_GMP
        mpz_import(dst, n_bytes, -1, sizeof(byte), -1, 0, src);
#else
        boost::multiprecision::cpp_int tmp;
        import_bits(tmp, src, src + n_bytes, CHAR_BIT, false);
        dst = tmp.backend();
#endif
    };

    const auto* src = raw_bytes.begin();
    for (dimn_t i = 0; i < count; ++i) {
        auto& item = raw[i];
        auto den = denominator(item);

#if RPY_USING_GMP
        auto* numref = mpq_numref(item.backend().data());
        auto* denref = mpq_denref(item.backend().data());
        unpack_limb(numref, src + pos);
        unpack_limb(denref, src + pos);
#else
        unpack_limb(item.backend().num(), src + pos);
        unpack_limb(item.backend().denom(), src + pos);
#endif
    }

    return out;
}
void RationalType::add_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l + r; }
    );
}
void RationalType::sub_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l - r; }
    );
}
void RationalType::mul_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l * r; }
    );
}
void RationalType::div_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) {
                if (r == decltype(r)(0)) {
                    RPY_THROW(std::runtime_error, "division by zero");
                }
                return l / r;
            }
    );
}
void RationalType::uminus_into(
        ScalarPointer& dst, const ScalarPointer& arg, dimn_t count,
        const uint64_t* mask
) const
{
    helper::unary_into_buffer(
            dst, arg, count, mask, [](auto s) { return -s; }
            );
}
