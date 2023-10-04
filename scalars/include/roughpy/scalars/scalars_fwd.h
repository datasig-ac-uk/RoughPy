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
// Created by user on 25/02/23.
//

#ifndef ROUGHPY_SCALARS_SCALARS_PREDEF_H
#define ROUGHPY_SCALARS_SCALARS_PREDEF_H

#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/device.h>
#include <roughpy/platform/serialization.h>

#include <complex>

#include <Eigen/Core>
#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/packed_integer.h>
#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/polynomial_basis.h>

#ifdef LAL_NO_USE_GMP
#  define RPY_USING_GMP 0
#else
#  define RPY_USING_GMP 1
#endif

#include <boost/multiprecision/cpp_int.hpp>

namespace rpy {
namespace scalars {

/// IEEE half-precision floating point type
using Eigen::half;

/// BFloat16 (truncated) floating point type
using Eigen::bfloat16;

/// Rational scalar type
using rational_scalar_type = lal::rational_field::scalar_type;

/// half-precision complex float
using half_complex = std::complex<half>;

/// BFloat16 complex float - probably not supported anywhere
using bf16_complex = std::complex<bfloat16>;

/// Single precision complex float
using float_complex = std::complex<float>;

/// double precision complex float
using double_complex = std::complex<double>;

/// Monomial key-type of polynomials
using monomial = lal::monomial;

/// Indeterminate type for monomials
using indeterminate_type = typename monomial::letter_type;

/// Polynomial (with rational coefficients) scalar type
using rational_poly_scalar = lal::polynomial<lal::rational_field>;

/// Marker for signed size type (ptrdiff_t)
struct signed_size_type_marker {
};

/// Marker for unsigned size type (size_t)
struct unsigned_size_type_marker {
};

/**
 * @brief Type codes for different scalar types.
 *
 * These are chosen to be compatible with the DLPack
 * array interchange protocol. Rational types will
 * be encoded as OpaqueHandle, since they're not simple
 * data. Some of these types might not be compatible with
 * this library.
 */
enum class ScalarTypeCode : uint8_t
{
    Int = 0U,
    UInt = 1U,
    Float = 2U,
    OpaqueHandle = 3U,
    BFloat = 4U,
    Complex = 5U,
    Bool = 6U
};

/**
 * @brief Basic information for identifying the type, size, and
 * configuration of a scalar.
 *
 * Based on, and compatible with, the DlDataType struct from the
 * DLPack array interchange protocol. The lanes parameter will
 * usually be set to 1, and is not generally used by RoughPy.
 */
struct BasicScalarInfo {
    ScalarTypeCode code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

/**
 * @brief A collection of basic information for identifying a scalar type.
 */
struct ScalarTypeInfo {
    string name;
    string id;
    size_t n_bytes;
    size_t alignment;
    BasicScalarInfo basic_info;
    platform::DeviceInfo device;
};

// Forward declarations

class ScalarType;

class ScalarInterface;

class ScalarPointer;

class Scalar;

class ScalarArray;

class OwnedScalarArray;

class KeyScalarArray;

class ScalarStream;

class RandomGenerator;

class BlasInterface;

template <typename T>
inline remove_cv_ref_t<T> scalar_cast(const Scalar& arg);

using conversion_function
        = std::function<void(ScalarPointer, ScalarPointer, dimn_t)>;

constexpr bool
operator==(const BasicScalarInfo& lhs, const BasicScalarInfo& rhs) noexcept
{
    return lhs.code == rhs.code && lhs.bits == rhs.bits
            && lhs.lanes == rhs.lanes;
}

/**
 * @brief Register a new type with the scalar type system
 * @param type Pointer to newly created ScalarType
 *
 *
 */
RPY_EXPORT void register_type(const ScalarType* type);

/**
 * @brief Get a type registered with the scalar type system
 * @param id Id string of type to be retrieved
 * @return pointer to ScalarType representing id
 */
RPY_EXPORT const ScalarType* get_type(const string& id);

RPY_EXPORT const ScalarType*
get_type(const string& id, const platform::DeviceInfo& device);

/**
 * @brief Get a list of all registered ScalarTypes
 * @return vector of ScalarType pointers.
 */
RPY_NO_DISCARD RPY_EXPORT std::vector<const ScalarType*> list_types();

RPY_NO_DISCARD RPY_EXPORT const ScalarTypeInfo& get_scalar_info(string_view id);

RPY_NO_DISCARD RPY_EXPORT const std::string&
id_from_basic_info(const BasicScalarInfo& info);

RPY_NO_DISCARD RPY_EXPORT const conversion_function&
get_conversion(const string& src_id, const string& dst_id);

RPY_EXPORT void register_conversion(
        const string& src_id,
        const string& dst_id,
        conversion_function converter
);

}// namespace scalars
}// namespace rpy

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
    MPIntegerSerializationHelper(Integer* p) : ptr(p) {}
    MPIntegerSerializationHelper(Integer& p) : ptr(&p) {}

private:
    using ptr_t = conditional_t<is_const<Integer>::value, const char*, char*>;

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
        ptr->reszie(new_size, new_size);
        return ptr->limbs();
#endif
    }

    void finalize(dimn_t size, bool isneg) noexcept
    {
#if RPY_USING_GMP
        mpz_limbs_finish(
                ptr,
                isneg ? -static_cast<mp_size_t>(size)
                      : static_cast<mp_size_t>(size)
        );
#else
        ptr->sign(isneg);
#endif
    }

    dimn_t nbytes() const noexcept { return size() * sizeof(limbs_t); }

public:
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
        RPY_SERIAL_SERIALIZE_SIZE(size);

        if (size > 0) {
            auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
            RPY_SERIAL_SERIALIZE_BYTES("data", resize(n_limbs), size);
            finalize(n_limbs, is_negative);
        }
    }

    void save(cereal::JSONOutputArchive& archive, const std::uint32_t RPY_UNUSED_VAR version) const {
        RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
        RPY_SERIAL_SERIALIZE_SIZE(nbytes());
        archive.saveBinaryValue(limbs(), nbytes(), "data");
    }

    void load(cereal::JSONInputArchive& archive, const std::uint32_t RPY_UNUSED_VAR version) {
        bool is_negative;
        dimn_t size;

        RPY_SERIAL_SERIALIZE_VAL(is_negative);
        RPY_SERIAL_SERIALIZE_SIZE(size);

        if (size > 0) {
            auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
            archive.loadBinaryValue(resize(n_limbs), size, "data");
            finalize(n_limbs, is_negative);
        }
    }

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

    RPY_SERIAL_SERIALIZE_SIZE(value.type());

    for (const auto& entry : value) { RPY_SERIAL_SERIALIZE_VAL(entry); }
}
RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::monomial)
{
    using namespace ::rpy;
    using namespace ::rpy::scalars;

    size_t count;
    RPY_SERIAL_SERIALIZE_SIZE(count);

    pair<indeterminate_type, deg_t> entry(indeterminate_type(0, 0), 0);
    for (size_t i = 0; i < count; ++i) {
        RPY_SERIAL_SERIALIZE_VAL(entry);
        value[entry.first] = entry.second;
    }
}

namespace cereal {
RPY_SERIAL_LOAD_FN_EXT(::rpy::scalars::rational_poly_scalar)
{
    using namespace rpy;
    using namespace rpy::scalars;

    size_t count;
    RPY_SERIAL_SERIALIZE_SIZE(count);

    monomial m;
    rational_scalar_type s;

    for (size_t i = 0; i < count; ++i) {
        RPY_SERIAL_SERIALIZE_BARE(m);
        rpy::scalars::dtl::load_rational(archive, s);
        value[m] = s;
    }
}

RPY_SERIAL_SAVE_FN_EXT(::rpy::scalars::rational_poly_scalar)
{
    using namespace rpy;
    using namespace rpy::scalars;

    RPY_SERIAL_SERIALIZE_SIZE(value.size());

    for (const auto& item : value) {
        RPY_SERIAL_SERIALIZE_BARE(item.key());
        rpy::scalars::dtl::save_rational(archive, item.value());
    }
}
}
#endif// ROUGHPY_SCALARS_SCALARS_PREDEF_H
