//
// Created by sam on 3/26/24.
//

#ifndef ARBITRARY_PRECISION_RATIONAL_H
#define ARBITRARY_PRECISION_RATIONAL_H

#include <roughpy/devices/core.h>
#include <boost/multiprecision/gmp.hpp>

#include <roughpy/platform/serialization.h>

#define RPY_USING_GMP 1

namespace rpy {
namespace scalars {

using ArbitraryPrecisionRational = boost::multiprecision::mpq_rational;

// template <>
// struct type_id_of_impl<ArbitraryPrecisionRational>
// {
//     static const string& get_id() noexcept;
// };

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
template <typename Archive>
void load_int(Archive& archive, mpz_ptr value)
{
    bool is_negative;
    rpy::dimn_t size;

    RPY_SERIAL_SERIALIZE_VAL(is_negative);
    {
        cereal::size_type _size;
        RPY_SERIAL_SERIALIZE_NVP("size", _size);
        size = static_cast<rpy::dimn_t>(_size);
    }

    rpy::dimn_t n_limbs = 0;
    if (size > 0) {
        n_limbs = (size + sizeof(mp_limb_t) - 1) / sizeof(mp_limb_t);
        RPY_SERIAL_SERIALIZE_BYTES(
                "data",
                mpz_limbs_write(value, static_cast<mp_size_t>(n_limbs)),
                size
        );
    }
    mpz_limbs_finish(
            value,
            is_negative ? -static_cast<mp_size_t>(n_limbs)
                        : static_cast<mp_size_t>(n_limbs)
    );
}

template <typename Archive>
void save_int(Archive& archive, mpz_srcptr value)
{
    auto is_negative = static_cast<unsigned char>(mpz_sgn(value));
    RPY_SERIAL_SERIALIZE_VAL(is_negative);
    const auto size = static_cast<cereal::size_type>(mpz_size(value));
    auto nbytes = size * sizeof(mp_limb_t);
    RPY_SERIAL_SERIALIZE_VAL(nbytes);
    RPY_SERIAL_SERIALIZE_BYTES("data", mpz_limbs_read(value), nbytes);
}

}// namespace scalars

namespace devices {

namespace dtl {
template <>
struct type_code_of_impl<scalars::ArbitraryPrecisionRational> {
    static constexpr TypeCode value = TypeCode::ArbitraryPrecisionRational;
};
template <>
struct type_id_of_impl<scalars::ArbitraryPrecisionRational>
{
    static constexpr string_view value = "Rational";
};

}// namespace dtl
}// namespace devices

}// namespace rpy

// namespace cereal {
//
// RPY_SERIAL_LOAD_FN_EXT(rpy::scalars::ArbitraryPrecisionRational)
// {
//     auto& backend = value.backend();
//     rpy::scalars::load_int(archive, mpq_numref(backend.data()));
//     rpy::scalars::load_int(archive, mpq_denref(backend.data()));
// }
//
// RPY_SERIAL_SAVE_FN_EXT(rpy::scalars::ArbitraryPrecisionRational)
// {
//     auto& backend = value.backend();
//     rpy::scalars::save_int(archive, mpq_numref(backend.data()));
//     rpy::scalars::save_int(archive, mpq_denref(backend.data()));
// }
// }// namespace cereal

#endif// ARBITRARY_PRECISION_RATIONAL_H
