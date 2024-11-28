//
// Created by sammorley on 25/11/24.
//

#include "rational_conversion.h"

#include <limits>

#include <gmp.h>
#include <mpfr.h>
#include <boost/container/flat_map.hpp>

#include "generics/builtin_types/builtin_type_ids.h"

#include "rational_type.h"
#include "multiprecision_type_ids.h"


using namespace rpy;
using namespace rpy::generics;

namespace {


template <typename T>
struct MPRationalFunctions
{
    static int set(mpq_ptr rop, const T* op) noexcept;

    static int get(T* rop, mpq_srcptr op) noexcept;

    static bool compare(mpq_srcptr lhs, const T* rhs) noexcept;
};



template <>
struct MPRationalFunctions<int8_t>
{
    static int set(mpq_ptr rop, const int8_t* op) noexcept {
        mpq_set_si(rop, *op, 1);
        return 0;
    }

    static int get(int8_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<int8_t>(mpz_get_si(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const int8_t* rhs) noexcept {
        return mpq_cmp_si(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<uint8_t>
{
    static int set(mpq_ptr rop, const uint8_t* op) noexcept {
        mpq_set_ui(rop, *op, 1);
        return 0;
    }

    static int get(uint8_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<uint8_t>(mpz_get_ui(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const uint8_t* rhs) noexcept {
        return mpq_cmp_ui(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<int16_t>
{
    static int set(mpq_ptr rop, const int16_t* op) noexcept {
        mpq_set_si(rop, *op, 1);
        return 0;
    }

    static int get(int16_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<int16_t>(mpz_get_si(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const int16_t* rhs) noexcept {
        return mpq_cmp_si(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<uint16_t>
{
    static int set(mpq_ptr rop, const uint16_t* op) noexcept {
        mpq_set_ui(rop, *op, 1);
        return 0;
    }

    static int get(uint16_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<uint16_t>(mpz_get_ui(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const uint16_t* rhs) noexcept {
        return mpq_cmp_ui(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<int32_t>
{
    static int set(mpq_ptr rop, const int32_t* op) noexcept {
        mpq_set_si(rop, *op, 1);
        return 0;
    }

    static int get(int32_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<int32_t>(mpz_get_si(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const int32_t* rhs) noexcept {
        return mpq_cmp_si(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<uint32_t>
{
    static int set(mpq_ptr rop, const uint32_t* op) noexcept {
        mpq_set_ui(rop, *op, 1);
        return 0;
    }

    static int get(uint32_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<uint32_t>(mpz_get_ui(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const uint32_t* rhs) noexcept {
        return mpq_cmp_ui(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<int64_t>
{
    static int set(mpq_ptr rop, const int64_t* op) noexcept {
        mpq_set_si(rop, *op, 1);
        return 0;
    }

    static int get(int64_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<int64_t>(mpz_get_si(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const int64_t* rhs) noexcept {
        return mpq_cmp_si(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<uint64_t>
{
    static int set(mpq_ptr rop, const uint64_t* op) noexcept {
        mpq_set_ui(rop, *op, 1);
        return 0;
    }

    static int get(uint64_t* rop, mpq_srcptr op) noexcept {
        if (mpz_cmp_si(mpq_denref(op), 1) != 0) {
            return 1;
        }
        *rop = static_cast<uint64_t>(mpz_get_ui(mpq_numref(op)));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const uint64_t* rhs) noexcept {
        return mpq_cmp_ui(lhs, *rhs, 1) == 0;
    }
};

template <>
struct MPRationalFunctions<float>
{
    static int set(mpq_ptr rop, const float* op) noexcept {
        mpq_set_d(rop, static_cast<double>(*op));
        return 0;
    }

    static int get(float* rop, mpq_srcptr op) noexcept {
        *rop = static_cast<float>(mpq_get_d(op));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const float* rhs) noexcept {
        return false;
    }
};

template <>
struct MPRationalFunctions<double>
{
    static int set(mpq_ptr rop, const double* op) noexcept {
        mpq_set_d(rop, *op);
        return 0;
    }

    static int get(double* rop, mpq_srcptr op) noexcept {
        *rop = mpq_get_d(op);
        return 0;
    }

    static bool compare(mpq_srcptr lhs, const double* rhs) noexcept {
        return false;
    }
};

template <>
struct MPRationalFunctions<MPInt> {

    static int set(mpq_ptr rop, mpz_srcptr op) noexcept
    {
        mpq_set_z(rop, op);
        return 0;
    }

    static int get(mpz_ptr rop, mpq_srcptr op) noexcept
    {
        auto* denom = mpq_denref(op);
        if (mpz_cmp_si(denom, 1) != 0) {
            return 1;
        }
        mpz_set(rop, mpq_numref(op));
        return 0;
    }

    static bool compare(mpq_srcptr lhs, mpz_srcptr rhs) noexcept
    {
        auto* denom = mpq_denref(lhs);
        if (mpz_cmp_si(denom, 1) != 0) {
            return false;
        }
        return mpz_cmp(mpq_numref(lhs), rhs) == 0;
    }
};


template <>
struct MPRationalFunctions<MPFloat>
{
    static int set(mpq_ptr rop, mpfr_srcptr op) noexcept
    {
        mpfr_get_q(rop, op);
        return 0;
    }

    static int get(mpfr_ptr rop, mpq_srcptr op) noexcept
    {
        mpfr_set_q(rop, op, MPFR_RNDN);
        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        return t;
    }

    static bool compare(mpq_srcptr lhs, mpfr_srcptr rhs) noexcept
    {
        return mpfr_cmp_q(rhs, lhs) == 0;
    }
};


template <typename T>
class FromMPRational : public ConversionTrait
{
public:
    FromMPRational(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename T>
bool FromMPRational<T>::is_exact() const noexcept
{
    return false;
}

template <typename T>
void FromMPRational<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<T*>(dst);
    const auto* src_ptr = static_cast<mpq_srcptr>(src);

    auto ret = MPRationalFunctions<T>::set(dst_ptr, src_ptr);
    if (ret > 0) {
        RPY_THROW(std::runtime_error, "failed to convert");
    }
    if (exact && ret < 0) {
        RPY_THROW(std::runtime_error, "non-exact conversion");
    }
}

template <typename T>
class ToMPRational : public ConversionTrait
{
public:
    ToMPRational(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;

};


template <typename T>
bool ToMPRational<T>::is_exact() const noexcept
{
    return true;
}

template <typename T>
void ToMPRational<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<mpq_ptr>(dst);
    const auto* src_ptr = static_cast<const T*>(src);

    auto ret = MPRationalFunctions<T>::get(dst_ptr, src_ptr);
    if (ret > 0) {
        RPY_THROW(std::runtime_error, "failed to convert");
    }
}

template <typename T>
class FromMPRationalFactory : public MPRationalConversionFromFactory
{
public:
    RPY_NO_DISCARD
    std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait>
FromMPRationalFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<MPRational>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<T>);

    return std::make_unique<FromMPRational<T>>(std::move(from_type), std::move(to_type));
}

template <typename T>
class ToMPRationalFactory : public MPRationalConversionToFactory
{
public:
    RPY_NO_DISCARD
    std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait>
ToMPRationalFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<T>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<MPRational>);
    return std::make_unique<ToMPRational<T>>(std::move(from_type), std::move(to_type));
}

}

#define ADD_FROM_FACTORY(Tp) \
    {hasher(type_id_of<Tp>), std::make_unique<FromMPRationalFactory<Tp>>()}


const MPRationalConversionFromFactory* MPRationalConversionFromFactory::
get_factory(const Type& type) noexcept
{
    using factory = std::unique_ptr<const MPRationalConversionFromFactory>;
    constexpr Hash<string_view> hasher;
    static const boost::container::flat_map<hash_t, factory> cache {
        ADD_FROM_FACTORY(double),
        ADD_FROM_FACTORY(float),
        ADD_FROM_FACTORY(int8_t),
        ADD_FROM_FACTORY(int16_t),
        ADD_FROM_FACTORY(int32_t),
        ADD_FROM_FACTORY(int64_t),
        ADD_FROM_FACTORY(uint8_t),
        ADD_FROM_FACTORY(uint16_t),
        ADD_FROM_FACTORY(uint32_t),
        ADD_FROM_FACTORY(uint64_t),
        ADD_FROM_FACTORY(MPRational),
        ADD_FROM_FACTORY(MPFloat)
    };

    if (auto it = cache.find(type.id()); it != cache.end()) {
        return it->second.get();
    }

    return nullptr;
}

#undef ADD_FROM_FACTORY

#define ADD_TO_FACTORY(Tp) \
    {hasher(type_id_of<Tp>), std::make_unique<ToMPRationalFactory<Tp>>()}

const MPRationalConversionToFactory* MPRationalConversionToFactory::get_factory(
    const Type& type) noexcept
{
    using factory = std::unique_ptr<const MPRationalConversionToFactory>;
    constexpr Hash<string_view> hasher;
    static const boost::container::flat_map<hash_t, factory> cache {
        ADD_TO_FACTORY(double),
        ADD_TO_FACTORY(float),
        ADD_TO_FACTORY(int8_t),
        ADD_TO_FACTORY(int16_t),
        ADD_TO_FACTORY(int32_t),
        ADD_TO_FACTORY(int64_t),
        ADD_TO_FACTORY(uint8_t),
        ADD_TO_FACTORY(uint16_t),
        ADD_TO_FACTORY(uint32_t),
        ADD_TO_FACTORY(uint64_t),
        ADD_TO_FACTORY(MPRational),
        ADD_TO_FACTORY(MPFloat)
    };

    if (auto it = cache.find(type.id()); it != cache.end()) {
        return it->second.get();
    }

    return nullptr;
}
