//
// Created by sammorley on 25/11/24.
//

#include "float_type.h"

#include "float_conversion.h"
#include "mpz_hash.h"


#include <gmp.h>
#include <mpfr.h>

using namespace rpy;
using namespace rpy::generics;

MPFloatType::MPFloatType(int precision)
    : m_arithmetic(this),
      m_comparison(this),
      m_number(this),
      m_precision(precision) { RPY_CHECK_LE(m_precision, MPFR_PREC_MAX); }

string_view MPFloatType::name() const noexcept { return "MultiPrecisionFloat"; }

string_view MPFloatType::id() const noexcept { return "apf"; }

const std::type_info& MPFloatType::type_info() const noexcept
{
    return typeid(mpfr_t);
}

BasicProperties MPFloatType::basic_properties() const noexcept
{
    return {
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            false
    };
}

size_t MPFloatType::object_size() const noexcept { return sizeof(mpfr_ptr); }

void* MPFloatType::allocate_object() const
{
    auto* new_float = static_cast<mpfr_ptr>(mem::aligned_alloc(
        alignof(mpfr_t),
        sizeof(mpfr_t)));

    if (new_float == nullptr) { throw std::bad_alloc(); }
    mpfr_init2(new_float, m_precision);
    mpfr_set_zero(new_float, 1);
    return new_float;
}

void MPFloatType::free_object(void* ptr) const
{
    RPY_DBG_ASSERT_NE(ptr, nullptr);
    mpfr_clear(static_cast<mpfr_ptr>(ptr));
    mem::aligned_free(ptr);
}

bool MPFloatType::parse_from_string(void* data, string_view str) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    string tmp(str);
    return mpfr_set_str(static_cast<mpfr_ptr>(data), tmp.c_str(), 10, MPFR_RNDN)
            != -1;
}

void MPFloatType::copy_or_fill(void* dst,
                               const void* src,
                               size_t count,
                               bool uninit) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);

    if (RPY_UNLIKELY(count == 0)) { return; }

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);
    if (src == nullptr) {
        if (uninit) {
            for (size_t i = 0; i < count; ++i, ++dst_ptr) {
                mpfr_init2(dst_ptr, m_precision);
                mpfr_set_zero(dst_ptr, 1);
            }
        } else {
            for (size_t i = 0; i < count; ++i) { mpfr_set_zero(++dst_ptr, 1); }
        }
    } else {
        const auto* src_ptr = static_cast<mpfr_srcptr>(src);
        if (uninit) {
            for (size_t i = 0; i < count; ++i, ++dst_ptr, ++src_ptr) {
                mpfr_init2(dst_ptr, m_precision);
                mpfr_set(dst_ptr, src_ptr, MPFR_RNDN);
            }
        } else {
            for (size_t i = 0; i < count; ++i, ++src_ptr, ++dst_ptr) {
                mpfr_set(dst_ptr, src_ptr, MPFR_RNDN);
            }
        }
    }

}

void MPFloatType::move(void* dst, void* src, size_t count, bool uninit) const
{
    if (RPY_UNLIKELY(count == 0)) { return; }

    RPY_DBG_ASSERT_NE(src, nullptr);
    RPY_DBG_ASSERT_NE(dst, nullptr);

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);
    auto* src_ptr = static_cast<mpfr_ptr>(src);

    const auto prec = mpfr_get_prec(src_ptr);
    RPY_CHECK_EQ(prec, m_precision);

    if (uninit) {
        for (size_t i = 0; i < count; ++i, ++dst_ptr, ++src_ptr) {
            mpfr_init2(dst_ptr, m_precision);
            mpfr_swap(dst_ptr, src_ptr);
        }
    } else {
        for (size_t i = 0; i < count; ++i, ++dst_ptr, ++src_ptr) {
            mpfr_swap(dst_ptr, src_ptr);
        }
    }
}

void MPFloatType::destroy_range(void* data, size_t count) const
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* ptr = static_cast<mpfr_ptr>(data);
    for (size_t i = 0; i < count; ++i) { mpfr_clear(ptr++); }
}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_to(const Type& type) const noexcept
{
    static const auto table = conv::make_mpfloat_conversion_to_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(this, &type);
    }

    return Type::convert_to(type);
}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_from(const Type& type) const noexcept
{
    static const auto table = conv::make_mpfloat_conversion_from_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(this, &type);
    }

    return Type::convert_to(type);
}

const BuiltinTrait* MPFloatType::
get_builtin_trait(BuiltinTraitID id) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Comparison: return &m_comparison;
        case BuiltinTraitID::Arithmetic: return &m_arithmetic;
        case BuiltinTraitID::Number: return &m_number;
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

namespace {

struct StringCleanup
{
    char* ptr = nullptr;

    ~StringCleanup() { if (ptr != nullptr) { mpfr_free_str(ptr); } }
};


}

const std::ostream& MPFloatType::display(std::ostream& os,
                                         const void* value) const
{
    if (value == nullptr) { return os << 0; }

    const auto* flt_val = static_cast<mpfr_srcptr>(value);

    // Choosing a format to emulate double output.
    // "%.17g" is a common format to represent doubles with sufficient precision
    // where 17 significant digits should be sufficient to represent a double value.
    StringCleanup buffer;
    const auto n_digits = mpfr_asprintf(&buffer.ptr, "%.17Rg", flt_val);

    RPY_CHECK_GT(n_digits, 0);

    return os << string_view(buffer.ptr, n_digits);
}


namespace {

struct MPZCleanup
{
    mpz_t value;

    ~MPZCleanup()
    {
        mpz_clear(value);
    }
};

}

hash_t MPFloatType::hash_of(const void* value) const noexcept
{
    if (value == nullptr) { return 0; }

    const auto* flt_val = static_cast<mpfr_srcptr>(value);

    MPZCleanup significand;
    auto exp = mpfr_get_z_2exp(significand.value, flt_val);

    auto result = mpz_hash(significand.value);
    hash_combine(result, exp);

    return result;
}