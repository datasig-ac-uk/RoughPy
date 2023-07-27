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

#include <roughpy/scalars/conversion.h>
#include <roughpy/scalars/random.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_type_helper.h>

#include "standard_random_generator.h"

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
class StandardScalarType : public ScalarType
{

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
        : ScalarType({
                std::move(name),
                std::move(id),
                sizeof(ScalarImpl),
                alignof(ScalarImpl),
                {ScalarTypeCode::Float, sizeof_bits<ScalarImpl>(), 1U},
                {ScalarDeviceType::CPU, 0}
    })
    {}

    explicit StandardScalarType(const ScalarTypeInfo& info)
        : ScalarType(ScalarTypeInfo(info))
    {}

    explicit StandardScalarType(
            string name, string id, std::size_t size, std::size_t align,
            BasicScalarInfo basic_info, ScalarDeviceInfo device_info
    )
        : ScalarType({name, id, size, align, basic_info, device_info})
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
                    this, new ScalarImpl, flags::IsMutable | flags::OwnedPointer
            );
        } else {
            return ScalarPointer(
                    this, new ScalarImpl[size],
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

    void swap(ScalarPointer lhs, ScalarPointer rhs) const override
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
            return lhs.type()->swap(lhs, rhs);
        }

        if (lhs.is_const() || rhs.is_const()) {
            RPY_THROW(
                    std::runtime_error, "one or both of the scalars is const"
            );
        }

        std::swap(*lhs.raw_cast<ScalarImpl*>(), *rhs.raw_cast<ScalarImpl*>());
    }

protected:
    ScalarImpl try_convert(ScalarPointer other) const
    {
        if (other.is_null()) { return ScalarImpl(0); }
        if (other.type() == this) {
            return *other.template raw_cast<const ScalarImpl>();
        }

        const ScalarType* type = other.type();
        if (type == nullptr) {
            RPY_THROW(std::runtime_error, "null type for non-zero value");
        }

        auto cv = get_conversion(type->id(), this->id());
        if (cv) {
            ScalarImpl result;
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

public:
    void convert_copy(void* out, ScalarPointer in, dimn_t count) const override
    {
        RPY_DBG_ASSERT(out != nullptr);
        RPY_DBG_ASSERT(!in.is_null());
        const auto* type = in.type();

        if (type == nullptr) {
            RPY_THROW(std::runtime_error, "null type for non-zero value");
        }

        if (type == this) {
            const auto* in_begin = in.template raw_cast<const ScalarImpl>();
            const auto* in_end = in_begin + count;
            std::copy(in_begin, in_end, static_cast<ScalarImpl*>(out));
        } else {
            const auto& cv = get_conversion(type->id(), this->id());
            ScalarPointer out_ptr{this, out};

            cv(out_ptr, in, count);
        }
    }

private:
    template <typename Basic>
    void convert_copy_basic(ScalarPointer& out, const void* in, dimn_t count)
            const noexcept
    {
        const auto* iptr = static_cast<const Basic*>(in);
        auto* optr = static_cast<ScalarImpl*>(out.ptr());

        for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
            ::new (optr) ScalarImpl(*iptr);
        }
    }

public:
    void convert_copy(
            ScalarPointer out, const void* in, dimn_t count,
            const string& type_id
    ) const override
    {
        if (type_id == "f64") {
            return convert_copy_basic<double>(out, in, count);
        } else if (type_id == "f32") {
            return convert_copy_basic<float>(out, in, count);
        } else if (type_id == "i32") {
            return convert_copy_basic<int>(out, in, count);
        } else if (type_id == "u32") {
            return convert_copy_basic<unsigned int>(out, in, count);
        } else if (type_id == "i64") {
            return convert_copy_basic<long long>(out, in, count);
        } else if (type_id == "u64") {
            return convert_copy_basic<unsigned long long>(out, in, count);
        } else if (type_id == "isize") {
            return convert_copy_basic<std::ptrdiff_t>(out, in, count);
        } else if (type_id == "usize") {
            return convert_copy_basic<std::size_t>(out, in, count);
        } else if (type_id == "i16") {
            return convert_copy_basic<short>(out, in, count);
        } else if (type_id == "u16") {
            return convert_copy_basic<unsigned short>(out, in, count);
        } else if (type_id == "i8") {
            return convert_copy_basic<char>(out, in, count);
        } else if (type_id == "u8") {
            return convert_copy_basic<unsigned char>(out, in, count);
        }

        // If we're here, then it is a non-standard type
        const auto& conversion = get_conversion(type_id, this->id());
        conversion(out, {nullptr, in}, count);
    }

    void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count)
            const override
    {
        if (src.type() == nullptr) {
            if (!src.is_simple_integer()) {
                RPY_THROW(
                        std::runtime_error,
                        "no type associated with scalar value"
                );
            }
            switch (src.simple_integer_config()) {
                case flags::UnsignedInteger8:
                    convert_copy_basic<uint8_t>(dst, src.cptr(), count);
                    break;
                case flags::UnsignedInteger16:
                    convert_copy_basic<uint16_t>(dst, src.cptr(), count);
                    break;
                case flags::UnsignedInteger32:
                    convert_copy_basic<uint32_t>(dst, src.cptr(), count);
                    break;
                case flags::UnsignedInteger64:
                    convert_copy_basic<uint64_t>(dst, src.cptr(), count);
                    break;
                case flags::UnsignedSize:
                    convert_copy_basic<dimn_t>(dst, src.cptr(), count);
                    break;
                case flags::SignedInteger8:
                    convert_copy_basic<int8_t>(dst, src.cptr(), count);
                    break;
                case flags::SignedInteger16:
                    convert_copy_basic<int16_t>(dst, src.cptr(), count);
                    break;
                case flags::SignedInteger32:
                    convert_copy_basic<int32_t>(dst, src.cptr(), count);
                    break;
                case flags::SignedInteger64:
                    convert_copy_basic<int64_t>(dst, src.cptr(), count);
                    break;
                case flags::SignedSize:
                    convert_copy_basic<idimn_t>(dst, src.cptr(), count);
                    break;
            }
        } else {
            convert_copy(dst, src.cptr(), count, src.type()->id());
        }
    }
    void convert_copy(
            void* out, const void* in, std::size_t count, BasicScalarInfo info
    ) const override
    {

        ScalarPointer optr(this, out);
        switch (info.code) {
            case ScalarTypeCode::Int:
                switch (info.bits) {
                    case sizeof(int8_t) * CHAR_BIT:
                        convert_copy_basic<int8_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(int16_t) * CHAR_BIT:
                        convert_copy_basic<int16_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(int32_t) * CHAR_BIT:
                        convert_copy_basic<int32_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(int64_t) * CHAR_BIT:
                        convert_copy_basic<int64_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                        //                    case 128:
                    default:
                        RPY_THROW(
                                std::runtime_error,
                                "invalid bit configuration for integer type"
                        );
                }
                break;
            case ScalarTypeCode::UInt:
                switch (info.bits) {
                    case sizeof(uint8_t) * CHAR_BIT:
                        convert_copy_basic<uint8_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(uint16_t) * CHAR_BIT:
                        convert_copy_basic<uint16_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(uint32_t) * CHAR_BIT:
                        convert_copy_basic<uint32_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                    case sizeof(uint64_t) * CHAR_BIT:
                        convert_copy_basic<uint64_t>(
                                optr, in, info.lanes * count
                        );
                        break;
                        //                    case 128:
                    default:
                        RPY_THROW(
                                std::runtime_error,
                                "invalid bit configuration for integer type"
                        );
                }
                break;
            case ScalarTypeCode::Float:
                switch (info.bits) {
                    case sizeof(half) * CHAR_BIT:
                        convert_copy_basic<half>(optr, in, info.lanes * count);
                        break;
                    case sizeof(float) * CHAR_BIT:
                        convert_copy_basic<float>(optr, in, info.lanes * count);
                        break;
                    case sizeof(double) * CHAR_BIT:
                        convert_copy_basic<double>(
                                optr, in, info.lanes * count
                        );
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
                    case sizeof(bfloat16) * CHAR_BIT:
                        convert_copy_basic<bfloat16>(
                                optr, in, info.lanes * count
                        );
                        break;
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

    Scalar copy(ScalarPointer arg) const override
    {
        return Scalar(this, try_convert(arg));
    }
    Scalar uminus(ScalarPointer arg) const override
    {
        return Scalar(this, -try_convert(arg));
    }
    Scalar add(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        if (!lhs) { return copy(rhs); }
        return Scalar(
                this, *lhs.raw_cast<const ScalarImpl*>() + try_convert(rhs)
        );
    }
    Scalar sub(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        if (!lhs) { return uminus(rhs); }
        return Scalar(
                this, *lhs.raw_cast<const ScalarImpl*>() - try_convert(rhs)
        );
    }
    Scalar mul(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        if (!lhs) { return zero(); }
        return Scalar(
                this, *lhs.raw_cast<const ScalarImpl*>() * try_convert(rhs)
        );
    }
    Scalar div(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        if (!lhs) { return zero(); }
        if (rhs.is_null()) {
            RPY_THROW(std::runtime_error, "division by zero");
        }

        auto crhs = try_convert(rhs);

        if (crhs == ScalarImpl(0)) {
            RPY_THROW(std::runtime_error, "division by zero");
        }

        return Scalar(
                this,
                static_cast<ScalarImpl>(
                        *lhs.raw_cast<const ScalarImpl*>() / crhs
                )
        );
    }

private:
    template <typename F>
    static void op_into_opt(
            ScalarImpl* RPY_RESTRICT dptr, const ScalarImpl* RPY_RESTRICT lptr,
            const ScalarImpl* RPY_RESTRICT rptr, dimn_t count,
            const uint64_t* mask, F&& func
    )
    {}

    template <typename F>
    static void
    op_into(ScalarPointer& dst, const ScalarPointer& lhs,
            const ScalarPointer& rhs, dimn_t count, const uint64_t* mask,
            F&& func);

public:
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
        return *lhs.raw_cast<const ScalarImpl*>() == try_convert(rhs);
    }

    Scalar one() const override { return Scalar(this, ScalarImpl(1)); }
    Scalar mone() const override { return Scalar(this, ScalarImpl(-1)); }
    Scalar zero() const override { return Scalar(this, ScalarImpl(0)); }
    void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        RPY_DBG_ASSERT(lhs);
        auto* ptr = lhs.raw_cast<ScalarImpl*>();
        *ptr += try_convert(rhs);
    }
    void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        RPY_DBG_ASSERT(lhs);
        auto* ptr = lhs.raw_cast<ScalarImpl*>();
        *ptr -= try_convert(rhs);
    }
    void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        RPY_DBG_ASSERT(lhs);
        auto* ptr = lhs.raw_cast<ScalarImpl*>();
        *ptr *= try_convert(rhs);
    }
    void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const override
    {
        RPY_DBG_ASSERT(lhs);
        auto* ptr = lhs.raw_cast<ScalarImpl*>();
        if (rhs.is_null()) {
            RPY_THROW(std::runtime_error, "division by zero");
        }

        auto crhs = try_convert(rhs);

        if (crhs == ScalarImpl(0)) {
            RPY_THROW(std::runtime_error, "division by zero");
        }

        *ptr /= crhs;
    }
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
template <typename F>
void StandardScalarType<ScalarImpl>::op_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask, F&& func
)
{}

template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::add_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<ScalarImpl>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l + r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::sub_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<ScalarImpl>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l - r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::mul_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<ScalarImpl>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l * r; }
    );
}
template <typename ScalarImpl>
void StandardScalarType<ScalarImpl>::div_into(
        ScalarPointer& dst, const ScalarPointer& lhs, const ScalarPointer& rhs,
        dimn_t count, const uint64_t* mask
) const
{
    impl_helpers::binary_into_buffer<ScalarImpl>(
            dst, lhs, rhs, count, mask, [](auto l, auto r) { return l + r; }
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
