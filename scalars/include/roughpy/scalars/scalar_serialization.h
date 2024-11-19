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

#ifndef ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_
#define ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_

#include <roughpy/core/helpers.h>
#include <roughpy/platform/serialization.h>

#include "scalar_types.h"
#include <cereal/types/utility.hpp>

RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::half)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;

    uint16_t tmp;
    RPY_SERIAL_SERIALIZE_NVP("value", tmp);
    value = bit_cast<half>(tmp);
}

RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::half)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;
    RPY_SERIAL_SERIALIZE_NVP("value", bit_cast<uint16_t>(value));
}

RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::bfloat16)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;

    uint16_t tmp;
    RPY_SERIAL_SERIALIZE_NVP("value", tmp);
    value = bit_cast<bfloat16>(tmp);
}

RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::bfloat16)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;
    RPY_SERIAL_SERIALIZE_NVP("value", bit_cast<uint16_t>(value));
}

/*
 * Here's the deal with rationals. Both GMP rationals and cpp_int rationals,
 * the two implementations available through libalgebra-lite, store a pair of
 * arbitrary precision integers as the numerator and denominator. The
 * integers are an array of "limbs" that store the bits of the magnitude in
 * little-endian order. The GMP implementation encodes the sign in the number
 * of limbs, while the cpp_int implementation has a separate bool member.
 *
 * To serialize we can just save the sign, number of bytes per limb, number
 * of limbs, and then the sequence of bytes from the limb array.
 *
 * To deserialize, we need to load the sign and number of bytes/limbs, and then
 * work out how many limbs need to be allocated to accommodate the integer.
 *
 * Unfortunately, because integer types are convertible to rationals, this
 * causes problems elsewhere in the library, so instead of implementing the
 * generic serialize methods, we're going to create our own methods that we can
 * use to implement the polynomial serialization.
 */

namespace rpy {
namespace scalars {
namespace dtl {

template <typename Integer>
class MPIntegerSerializationHelper
{
    Integer* ptr;

public:
    MPIntegerSerializationHelper(Integer* p)
        : ptr(p) {}

    MPIntegerSerializationHelper(Integer& p)
        : ptr(&p) {}

    using ptr_t = conditional_t<is_const_v<Integer>, const char*, char*>;

#if RPY_USING_GMP
    using limbs_t = mp_limb_t;
#else
    using limbs_t = boost::multiprecision::limb_type;
#endif

    const limbs_t* limbs() const noexcept
    {
#if RPY_USING_GMP
        return mpz_limbs_read(ptr);
#else
        return ptr->limbs();
#endif
    }

    bool is_negative() const noexcept
    {
#if RPY_USING_GMP
        return mpz_sgn(ptr) < 0;
#else
        return ptr->sign();
#endif
    }

    static constexpr dimn_t sizeof_limb() noexcept { return sizeof(limbs_t); }

    dimn_t size() const noexcept
    {
#if RPY_USING_GMP
        return static_cast<dimn_t>(mpz_size(ptr));
#else
        return ptr->size();
#endif
    }

    limbs_t* resize(dimn_t new_size) noexcept
    {
#if RPY_USING_GMP
        return mpz_limbs_write(ptr, static_cast<mp_size_t>(new_size));
#else
        ptr->resize(new_size, new_size);
        return ptr->limbs();
#endif
    }

    void finalize(dimn_t size, bool isneg) noexcept
    {
#if RPY_USING_GMP
        mpz_limbs_finish(
            ptr,
            isneg
            ? -static_cast<mp_size_t>(size)
            : static_cast<mp_size_t>(size)
        );
#else
        ptr->sign(isneg);
#endif
    }

    dimn_t nbytes() const noexcept { return size() * sizeof(limbs_t); }

    dimn_t total_bytes() const noexcept
    {
        return 1 + sizeof(uint64_t) + nbytes();
    }

    RPY_SERIAL_SAVE_FN()
    {
        RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
        RPY_SERIAL_SERIALIZE_SIZE(nbytes());
        RPY_SERIAL_SERIALIZE_BYTES("data", limbs(), nbytes());
    }

    RPY_SERIAL_LOAD_FN()
    {
        bool is_negative;
        dimn_t size;

        RPY_SERIAL_SERIALIZE_VAL(is_negative);
        {
            // This is probably redundant, but keep the type system happy.
            serial::size_type tmp_size;
            RPY_SERIAL_SERIALIZE_SIZE(tmp_size);
            size = static_cast<dimn_t>(tmp_size);
        }

        dimn_t n_limbs = 0;
        if (size > 0) {
            n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
            RPY_SERIAL_SERIALIZE_BYTES("data", resize(n_limbs), size);
        }
        finalize(n_limbs, is_negative);
    }

    void
    save(cereal::JSONOutputArchive& archive,
         const std::uint32_t RPY_UNUSED_VAR version) const;

    void
    load(cereal::JSONInputArchive& archive,
         const std::uint32_t RPY_UNUSED_VAR version);

    void
    save(cereal::XMLOutputArchive& archive,
         const std::uint32_t RPY_UNUSED_VAR version) const;

    void
    load(cereal::XMLInputArchive& archive,
         const std::uint32_t RPY_UNUSED_VAR version);
};

template <typename Archive>
void save_rational(Archive& archive, const rational_scalar_type& value)
{
    const auto& backend = value.backend();

#if RPY_USING_GMP
    using helper_t = MPIntegerSerializationHelper<remove_pointer_t<mpz_srcptr>>;

    RPY_SERIAL_SERIALIZE_NVP("numerator", helper_t(mpq_numref(backend.data())));
    RPY_SERIAL_SERIALIZE_NVP(
        "denominator",
        helper_t(mpq_denref(backend.data()))
    );
#else
    using helper_t = MPIntegerSerializationHelper<
            const boost::multiprecision::cpp_int_backend>;

    RPY_SERIAL_SERIALIZE_NVP("numerator", helper_t(backend.num()));
    RPY_SERIAL_SERIALIZE_NVP("denominator", helper_t(backend.den()));
#endif
}

template <typename Archive>
void load_rational(Archive& archive, rational_scalar_type& value)
{
    auto& backend = value.backend();

#if RPY_USING_GMP
    using helper_t = MPIntegerSerializationHelper<remove_pointer_t<mpz_ptr>>;

    RPY_SERIAL_SERIALIZE_NVP("numerator", helper_t(mpq_numref(backend.data())));
    RPY_SERIAL_SERIALIZE_NVP(
        "denominator",
        helper_t(mpq_denref(backend.data()))
    );
#else
    using helper_t = MPIntegerSerializationHelper<
            boost::multiprecision::cpp_int_backend>;

    RPY_SERIAL_SERIALIZE_NVP("numerator", helper_t(backend.num()));
    RPY_SERIAL_SERIALIZE_NVP("denominator", helper_t(backend.den()));
#endif
}

}// namespace dtl
}// namespace scalars
}// namespace rpy

RPY_SERIAL_EXT_LIB_LOAD_FN(rpy::scalars::indeterminate_type)
{
    using namespace ::rpy::scalars;
    using packed_type = typename indeterminate_type::packed_type;
    using integral_type = typename indeterminate_type::integral_type;

    packed_type symbol;
    RPY_SERIAL_SERIALIZE_VAL(symbol);

    integral_type index;
    RPY_SERIAL_SERIALIZE_VAL(index);

    value = indeterminate_type(symbol, index);
}

RPY_SERIAL_EXT_LIB_SAVE_FN(rpy::scalars::indeterminate_type)
{
    using namespace ::rpy::scalars;
    using packed_type = typename indeterminate_type::packed_type;
    using integral_type = typename indeterminate_type::integral_type;
    RPY_SERIAL_SERIALIZE_NVP("symbol", static_cast<packed_type>(value));
    RPY_SERIAL_SERIALIZE_NVP("index", static_cast<integral_type>(value));
}

RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::monomial)
{
    using namespace ::rpy::scalars;

    RPY_SERIAL_SERIALIZE_SIZE(static_cast<cereal::size_type>(value.type()));

    for (auto&& entry : value) {
        //        RPY_SERIAL_SERIALIZE_NVP("key", entry.first);
        //        RPY_SERIAL_SERIALIZE_NVP("degree", entry.second);
        RPY_SERIAL_SERIALIZE_BARE(
            ::cereal::make_map_item(entry.first, entry.second)
        );
        //        RPY_SERIAL_SERIALIZE_BARE(entry);
    }
}

RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::monomial)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;
    cereal::size_type count;
    RPY_SERIAL_SERIALIZE_SIZE(static_cast<cereal::size_type&>(count));

    //    pair<indeterminate_type, deg_t> entry(indeterminate_type(0, 0), 0);
    indeterminate_type id(0, 0);
    deg_t degree;
    for (size_t i = 0; i < count; ++i) {
        //        RPY_SERIAL_SERIALIZE_BARE(entry);
        RPY_SERIAL_SERIALIZE_BARE(::cereal::make_map_item(id, degree));

        value[id] = degree;
    }
}

// namespace cereal {
RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::rational_poly_scalar)
{
    using namespace rpy;
    using namespace rpy::scalars;

    cereal::size_type count;
    RPY_SERIAL_SERIALIZE_SIZE(count);

    monomial m;
    rational_scalar_type s;

    for (size_t i = 0; i < count; ++i) {
        RPY_SERIAL_SERIALIZE_BARE(m);
        rpy::scalars::dtl::load_rational(archive, s);
        value[m] = s;
    }
}

RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::rational_poly_scalar)
{
    using namespace rpy;
    using namespace rpy::scalars;

    RPY_SERIAL_SERIALIZE_SIZE(value.size());

    for (const auto& item : value) {
        RPY_SERIAL_SERIALIZE_BARE(item.key());
        rpy::scalars::dtl::save_rational(archive, item.value());
    }
}

//}// namespace cereal

#endif // ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_
