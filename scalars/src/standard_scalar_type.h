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
// Created by user on 02/03/23.
//

#ifndef ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H
#define ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H

#include <roughpy/scalars/scalar.h>

#include <limits>
#include <ostream>
#include <unordered_map>
#include <utility>

#include <roughpy/platform/device.h>
#include <roughpy/scalars/conversion.h>
#include <roughpy/scalars/random.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_type_helper.h>

#include "standard_random_generator.h"

using rpy::platform::DeviceType;
using rpy::platform::DeviceInfo;

namespace rpy {
namespace scalars {

template <typename T>
constexpr std::uint8_t sizeof_bits() noexcept
{
    return static_cast<std::uint8_t>(
            std::min(
                    static_cast<std::size_t>(
                            std::numeric_limits<std::uint8_t>::max() / CHAR_BIT
                    ),
                    sizeof(T)
            )
            * CHAR_BIT
    );
}

template <typename ScalarImpl>
class StandardScalarType : public impl_helpers::ScalarTypeHelper<ScalarImpl>
{
    using helper = impl_helpers::ScalarTypeHelper<ScalarImpl>;

    using rng_getter = std::unique_ptr<
            RandomGenerator> (*)(const ScalarType* type, Slice<uint64_t>);

    static std::unique_ptr<RandomGenerator>
    get_mt19937_generator(const ScalarType* type, Slice<uint64_t> seed);
    static std::unique_ptr<RandomGenerator>
    get_pcg_generator(const ScalarType* type, Slice<uint64_t> seed);

    std::unordered_map<string, rng_getter> m_rng_getters{
            {"mt19937", &get_mt19937_generator},
            {    "pcg",     &get_pcg_generator}
    };

public:
    explicit StandardScalarType(string id, string name)
        : helper({
                std::move(name),
                std::move(id),
                sizeof(ScalarImpl),
                alignof(ScalarImpl),
                {ScalarTypeCode::Float, sizeof_bits<ScalarImpl>(), 1U},
                {DeviceType::CPU, 0}
    })
    {}

    explicit StandardScalarType(const ScalarTypeInfo& info)
        : helper(ScalarTypeInfo(info))
    {}

    explicit StandardScalarType(
            string name, string id, std::size_t size, std::size_t align,
            BasicScalarInfo basic_info, DeviceInfo device_info
    )
        : helper({name, id, size, align, basic_info, device_info})
    {}

    Scalar
    from(long long int numerator, long long int denominator) const override
    {
        return Scalar(this, ScalarImpl(numerator) / ScalarImpl(denominator));
    }

    ScalarPointer allocate(dimn_t size) const override
    {
        if (size == 1) {
            return ScalarPointer(
                    this, new ScalarImpl {}, flags::IsMutable | flags::OwnedPointer
            );
        } else {
            return ScalarPointer(
                    this, new ScalarImpl[size] {},
                    flags::IsMutable | flags::OwnedPointer
            );
        }
    }
    void free(ScalarPointer pointer, dimn_t size) const override
    {
        if (!pointer.is_null()) {
            if (size == 1) {
                delete pointer.template raw_cast<ScalarImpl>();
            } else {
                delete[] pointer.template raw_cast<ScalarImpl>();
            }
        }
    }

    void swap(ScalarPointer lhs, ScalarPointer rhs, dimn_t count) const override
    {

        if (lhs.is_null() ^ rhs.is_null()) {
            RPY_THROW(std::runtime_error, "one of the pointers is null");
        }

        if (lhs.type() != rhs.type()) {
            RPY_THROW(
                    std::runtime_error, "cannot swap scalars of different types"
            );
        }

        if (lhs.type() != this && lhs.type() != nullptr) {
            return lhs.type()->swap(lhs, rhs, 0);
        }

        if (lhs.is_const() || rhs.is_const()) {
            RPY_THROW(
                    std::runtime_error, "one or both of the scalars is const"
            );
        }

        auto* lptr = lhs.raw_cast<ScalarImpl*>();
        auto* rptr = rhs.raw_cast<ScalarImpl*>();
        for (dimn_t i=0; i<count; ++i) {
            std::swap(lptr[i], rptr[i]);
        }

    }

    void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count)
            const override
    {
        helper::copy_convert(dst, src, count);
    }
    //    void convert_copy(
    //            void* out, const void* in, std::size_t count, BasicScalarInfo
    //            info
    //    ) const override
    //    {
    //
    //        ScalarPointer optr(this, out);
    //        switch (info.code) {
    //            case ScalarTypeCode::Int:
    //                switch (info.bits) {
    //                    case sizeof(int8_t) * CHAR_BIT:
    //                        convert_copy_basic<int8_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(int16_t) * CHAR_BIT:
    //                        convert_copy_basic<int16_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(int32_t) * CHAR_BIT:
    //                        convert_copy_basic<int32_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(int64_t) * CHAR_BIT:
    //                        convert_copy_basic<int64_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                        //                    case 128:
    //                    default:
    //                        RPY_THROW(
    //                                std::runtime_error,
    //                                "invalid bit configuration for integer
    //                                type"
    //                        );
    //                }
    //                break;
    //            case ScalarTypeCode::UInt:
    //                switch (info.bits) {
    //                    case sizeof(uint8_t) * CHAR_BIT:
    //                        convert_copy_basic<uint8_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(uint16_t) * CHAR_BIT:
    //                        convert_copy_basic<uint16_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(uint32_t) * CHAR_BIT:
    //                        convert_copy_basic<uint32_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    case sizeof(uint64_t) * CHAR_BIT:
    //                        convert_copy_basic<uint64_t>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                        //                    case 128:
    //                    default:
    //                        RPY_THROW(
    //                                std::runtime_error,
    //                                "invalid bit configuration for integer
    //                                type"
    //                        );
    //                }
    //                break;
    //            case ScalarTypeCode::Float:
    //                switch (info.bits) {
    //                    case sizeof(half) * CHAR_BIT:
    //                        convert_copy_basic<half>(optr, in, info.lanes *
    //                        count); break;
    //                    case sizeof(float) * CHAR_BIT:
    //                        convert_copy_basic<float>(optr, in, info.lanes *
    //                        count); break;
    //                    case sizeof(double) * CHAR_BIT:
    //                        convert_copy_basic<double>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    default:
    //                        RPY_THROW(
    //                                std::runtime_error,
    //                                "invalid bit configuration for float type"
    //                        );
    //                }
    //                break;
    //            case ScalarTypeCode::BFloat:
    //                switch (info.bits) {
    //                    case sizeof(bfloat16) * CHAR_BIT:
    //                        convert_copy_basic<bfloat16>(
    //                                optr, in, info.lanes * count
    //                        );
    //                        break;
    //                    default:
    //                        RPY_THROW(
    //                                std::runtime_error,
    //                                "invalid bit configuration for bfloat
    //                                type"
    //                        );
    //                }
    //                break;
    //            case ScalarTypeCode::Bool:
    //            case ScalarTypeCode::OpaqueHandle: break;
    //            case ScalarTypeCode::Complex:
    //            default: RPY_THROW(std::runtime_error, "unsupported scalar
    //            type");
    //        }
    //    }
    void
    assign(ScalarPointer target, long long int numerator,
           long long int denominator) const override
    {
        *target.raw_cast<ScalarImpl*>()
                = ScalarImpl(numerator) / ScalarImpl(denominator);
    }
    scalar_t to_scalar_t(ScalarPointer arg) const override
    {
        return static_cast<scalar_t>(*arg.raw_cast<const ScalarImpl*>());
    }


    void uminus_into(
            ScalarPointer& dst, const ScalarPointer& arg, dimn_t count,
            const uint64_t* mask
    ) const override;

    void add_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void sub_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void mul_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;
    void div_into(
            ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask
    ) const override;

    bool are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept override
    {
        return *lhs.raw_cast<const ScalarImpl*>()
                == helper::try_convert(rhs);
    }

    Scalar one() const override { return Scalar(this, ScalarImpl(1)); }
    Scalar mone() const override { return Scalar(this, ScalarImpl(-1)); }
    Scalar zero() const override { return Scalar(this, ScalarImpl(0)); }

    bool is_zero(ScalarPointer arg) const override
    {
        return !static_cast<bool>(arg)
                || *arg.raw_cast<const ScalarImpl*>() == ScalarImpl(0);
    }
    void print(ScalarPointer arg, std::ostream& os) const override
    {
        if (!arg) {
            os << 0.0;
        } else {
            os << *arg.raw_cast<const ScalarImpl*>();
        }
    }

    std::unique_ptr<RandomGenerator>
    get_rng(const string& bit_generator, Slice<uint64_t> seed) const override;

    std::vector<byte>
    to_raw_bytes(const ScalarPointer& ptr, dimn_t count) const override;
    ScalarPointer
    from_raw_bytes(Slice<byte> raw_bytes, dimn_t count) const override;
};

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::uminus_into(
        ScalarPointer& dst, const ScalarPointer& arg, dimn_t count,
        const uint64_t* mask
) const
{
    helper::unary_into_buffer(
            dst, arg, count, mask, [](auto s) { return -s; }
    );
}

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::add_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l + r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::sub_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l - r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::mul_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l * r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::div_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    helper::binary_into_buffer(
            dst, lhs, rhs, count, mask,
            [](auto l, auto r) {
                if (r == decltype(r)(0)) {
                    RPY_THROW(std::runtime_error, "division by zero");
                }
                return l / r;
            }
    );
}

template <typename ScalarImpl>
std::vector<byte> StandardScalarType<ScalarImpl>::to_raw_bytes(
        const ScalarPointer& ptr, dimn_t count
) const
{
    RPY_CHECK(ptr.type() == this);
    std::vector<byte> result(count * sizeof(ScalarImpl));

    std::copy_n(
            ptr.raw_cast<const byte*>(), count * sizeof(ScalarImpl),
            result.data()
    );

    return result;
}
template <typename ScalarImpl>
ScalarPointer StandardScalarType<ScalarImpl>::from_raw_bytes(
        Slice<byte> raw_bytes, dimn_t count
) const
{

    RPY_CHECK(count * sizeof(ScalarImpl) == raw_bytes.size());

    auto optr = allocate(count);
    auto* raw = optr.template raw_cast<char*>();

    std::copy(raw_bytes.begin(), raw_bytes.end(), raw);
    return optr;
}

inline uint64_t device_to_seed()
{
    std::random_device rd;
    uint64_t result(rd());
    result <<= 32;
    result += static_cast<uint64_t>(rd());
    return result;
}

template <typename ScalarImpl>
std::unique_ptr<RandomGenerator>
StandardScalarType<ScalarImpl>::get_mt19937_generator(
        const ScalarType* type, Slice<uint64_t> seed
)
{
    if (seed.empty()) {
        auto num_seed = device_to_seed();
        return std::unique_ptr<RandomGenerator>{
                new StandardRandomGenerator<ScalarImpl, std::mt19937_64>(
                        type, num_seed
                )};
    }
    return std::unique_ptr<RandomGenerator>{
            new StandardRandomGenerator<ScalarImpl, std::mt19937_64>(
                    type, seed
            )};
}
template <typename ScalarImpl>
std::unique_ptr<RandomGenerator>
StandardScalarType<ScalarImpl>::get_pcg_generator(
        const ScalarType* type, Slice<uint64_t> seed
)
{
    if (seed.empty()) {
        auto num_seed = device_to_seed();
        return std::unique_ptr<RandomGenerator>{
                new StandardRandomGenerator<ScalarImpl, pcg64>(type, num_seed)};
    }
    return std::unique_ptr<RandomGenerator>{
            new StandardRandomGenerator<ScalarImpl, pcg64>(type, seed)};
}

template <typename ScalarImpl>
std::unique_ptr<RandomGenerator> StandardScalarType<ScalarImpl>::get_rng(
        const string& bit_generator, Slice<uint64_t> seed
) const
{
    if (bit_generator.empty()) {
        return m_rng_getters.find("pcg")->second(this, seed);
    }

    auto found = m_rng_getters.find(bit_generator);
    if (found != m_rng_getters.end()) { return found->second(this, seed); }

    return ScalarType::get_rng(bit_generator, seed);
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H
