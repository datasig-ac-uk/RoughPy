//
// Created by sam on 28/11/24.
//

#include "integer_conversion.h"

#include <gmp.h>
#include <mpfr.h>

#include "generics/builtin_types/builtin_type_ids.h"

#include "integer_type.h"
#include "multiprecision_type_ids.h"


using namespace rpy;
using namespace rpy::generics;


namespace {

template <typename T>
struct MPIntFunctions
{
    static int set(mpz_ptr rop, const T* op) noexcept;

    static int get(T* rop, mpz_srcptr op) noexcept;

    static bool compare(mpz_srcptr lhs, const T* rhs) noexcept;
};


template <>
struct MPIntFunctions<int8_t>
{
    static int set(mpz_ptr rop, const int8_t* op) noexcept
    {
        mpz_set_si(rop, static_cast<long>(*op));
        return 0;
    }

    static int get(int8_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<int8_t>(mpz_get_si(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const int8_t* rhs) noexcept
    {
        return mpz_cmp_si(lhs, static_cast<long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<int16_t>
{
    static int set(mpz_ptr rop, const int16_t* op) noexcept
    {
        mpz_set_si(rop, static_cast<long>(*op));
        return 0;
    }

    static int get(int16_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<int16_t>(mpz_get_si(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const int16_t* rhs) noexcept
    {
        return mpz_cmp_si(lhs, static_cast<long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<int32_t>
{
    static int set(mpz_ptr rop, const int32_t* op) noexcept
    {
        mpz_set_si(rop, static_cast<long>(*op));
        return 0;
    }

    static int get(int32_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<int32_t>(mpz_get_si(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const int32_t* rhs) noexcept
    {
        return mpz_cmp_si(lhs, static_cast<long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<int64_t>
{
    static int set(mpz_ptr rop, const int64_t* op) noexcept
    {
        mpz_set_si(rop, *op);
        return 0;
    }

    static int get(int64_t* rop, mpz_srcptr op) noexcept
    {
        *rop = mpz_get_si(op);
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const int64_t* rhs) noexcept
    {
        return mpz_cmp_si(lhs, *rhs) == 0;
    }
};

template <>
struct MPIntFunctions<uint8_t>
{
    static int set(mpz_ptr rop, const uint8_t* op) noexcept
    {
        mpz_set_ui(rop, static_cast<unsigned long>(*op));
        return 0;
    }

    static int get(uint8_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<uint8_t>(mpz_get_ui(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const uint8_t* rhs) noexcept
    {
        return mpz_cmp_ui(lhs, static_cast<unsigned long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<uint16_t>
{
    static int set(mpz_ptr rop, const uint16_t* op) noexcept
    {
        mpz_set_ui(rop, static_cast<unsigned long>(*op));
        return 0;
    }

    static int get(uint16_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<uint16_t>(mpz_get_ui(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const uint16_t* rhs) noexcept
    {
        return mpz_cmp_ui(lhs, static_cast<unsigned long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<uint32_t>
{
    static int set(mpz_ptr rop, const uint32_t* op) noexcept
    {
        mpz_set_ui(rop, static_cast<unsigned long>(*op));
        return 0;
    }

    static int get(uint32_t* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<uint32_t>(mpz_get_ui(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const uint32_t* rhs) noexcept
    {
        return mpz_cmp_ui(lhs, static_cast<unsigned long>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<uint64_t>
{
    static int set(mpz_ptr rop, const uint64_t* op) noexcept
    {
        mpz_set_ui(rop, *op);
        return 0;
    }

    static int get(uint64_t* rop, mpz_srcptr op) noexcept
    {
        *rop = mpz_get_ui(op);
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const uint64_t* rhs) noexcept
    {
        return mpz_cmp_ui(lhs, *rhs) == 0;
    }
};

template <>
struct MPIntFunctions<float>
{
    static int set(mpz_ptr rop, const float* op) noexcept
    {
        mpz_set_d(rop, static_cast<double>(*op));
        return 0;
    }

    static int get(float* rop, mpz_srcptr op) noexcept
    {
        *rop = static_cast<float>(mpz_get_d(op));
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const float* rhs) noexcept
    {
        return mpz_cmp_d(lhs, static_cast<double>(*rhs)) == 0;
    }
};

template <>
struct MPIntFunctions<double>
{
    static int set(mpz_ptr rop, const double* op) noexcept
    {
        mpz_set_d(rop, *op);
        return 0;
    }

    static int get(double* rop, mpz_srcptr op) noexcept
    {
        *rop = mpz_get_d(op);
        return 0;
    }

    static bool compare(mpz_srcptr lhs, const double* rhs) noexcept
    {
        return mpz_cmp_d(lhs, *rhs) == 0;
    }
};

template <>
struct MPIntFunctions<MPFloat>
{
    static int set(mpz_ptr rop, mpfr_srcptr op) noexcept
    {
        return 0;
    }

    static int get(mpfr_ptr rop, mpz_srcptr op) noexcept
    {
        return mpfr_set_z(rop, op, MPFR_RNDN);
    }

    static bool compare(mpz_srcptr lhs, mpfr_srcptr rhs) noexcept
    {
        return mpfr_cmp_z(rhs, lhs) == 0;
    }
};

template <>
struct MPIntFunctions<MPRational>
{
    static int set(mpz_ptr rop, mpq_srcptr op) noexcept
    {
        mpz_set_q(rop, op);
        return 0;
    }
    static int get(mpq_ptr rop, mpz_srcptr op) noexcept
    {
        mpq_set_z(rop, op);
        return 0;
    }
    static bool compare(mpz_srcptr lhs, mpq_srcptr rhs) noexcept
    {
        return mpq_cmp_z(rhs, lhs) == 0;
    }
};



template <typename T>
class FromMPInt : public ConversionTrait
{
public:
    FromMPInt(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename T>
bool FromMPInt<T>::is_exact() const noexcept
{
    return false;
}

template <typename T>
void FromMPInt<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<T*>(dst);
    const auto* src_ptr = static_cast<mpz_srcptr>(src);

    using funcs = MPIntFunctions<T>;

    funcs::get(dst_ptr, src_ptr);
    if (!is_exact() && exact) {
        funcs::compare(src_ptr, dst_ptr);
    }
}

template <typename T>
class ToMPInt : public ConversionTrait
{
public:
    ToMPInt(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};


template <typename T>
bool ToMPInt<T>::is_exact() const noexcept
{
    if constexpr (std::is_integral_v<T>) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
void ToMPInt<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<mpz_srcptr>(dst);
    const auto* src_ptr = static_cast<const T*>(src);

    using funcs = MPIntFunctions<T>;

    funcs::set(dst_ptr, src_ptr);
    if (!is_exact() && exact) {
        funcs::compare(dst_ptr, src_ptr);
    }
}

template <typename T>
class FromMPIntFactory : public MPIntegerConversionFromFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait>
FromMPIntFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<T>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<MPInt>);
    return std::make_unique<FromMPInt<T>>(std::move(from_type), std::move(to_type));
}

template <typename T>
class ToMPIntFactory : public MPIntegerConversionToFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait>
ToMPIntFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<MPInt>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<T>);
    return std::make_unique<ToMPInt<T>>(std::move(from_type), std::move(to_type));
}

}


#define ADD_FROM_FACTORY(Tp) \
    {hasher(type_id_of<Tp>), std::make_unique<FromMPIntFactory<Tp>>()}


const MPIntegerConversionFromFactory* MPIntegerConversionFromFactory::
get_factory(const Type& type)
{
    using factory = std::unique_ptr<const MPIntegerConversionFromFactory>;
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
    {hasher(type_id_of<Tp>), std::make_unique<ToMPIntFactory<Tp>>()}

const MPIntegerConversionToFactory* MPIntegerConversionToFactory::get_factory(
    const Type& type)
{
    using factory = std::unique_ptr<const MPIntegerConversionToFactory>;
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


#undef ADD_TO_FACTORY