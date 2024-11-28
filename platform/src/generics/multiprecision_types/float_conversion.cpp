//
// Created by sam on 28/11/24.
//


#include "float_conversion.h"

#include <memory>

#include <gmp.h>
#include <mpfr.h>
#include <boost/container/flat_map.hpp>

#include "generics/builtin_types/builtin_type_ids.h"

#include "float_type.h"

#include "multiprecision_type_ids.h"


using namespace rpy;
using namespace rpy::generics;

namespace {

template <typename T>
struct TypeMPFRFunctions
{
    static int set(mpfr_ptr rop, T op, mpfr_rnd_t rnd) noexcept;
    static T get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept;
    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept;
};


// Specializations for integer types

// Unsigned 16-bit integer specialization
template <>
struct TypeMPFRFunctions<uint8_t>
{
    static int set(mpfr_ptr rop, uint8_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_ui(rop, op, rnd);
    }

    static uint8_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<uint8_t>(mpfr_get_ui(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_ushort_p(op, rnd) == 0;
    }
};

// Signed 16-bit integer specialization
template <>
struct TypeMPFRFunctions<int8_t>
{
    static int set(mpfr_ptr rop, int8_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_si(rop, op, rnd);
    }

    static int8_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<int8_t>(mpfr_get_si(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_sshort_p(op, rnd) == 0;
    }
};

// Unsigned 16-bit integer specialization
template <>
struct TypeMPFRFunctions<uint16_t>
{
    static int set(mpfr_ptr rop, uint16_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_ui(rop, op, rnd);
    }

    static uint16_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<uint16_t>(mpfr_get_ui(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_ushort_p(op, rnd) == 0;
    }
};

// Signed 16-bit integer specialization
template <>
struct TypeMPFRFunctions<int16_t>
{
    static int set(mpfr_ptr rop, int16_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_si(rop, op, rnd);
    }

    static int16_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<int16_t>(mpfr_get_si(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_sshort_p(op, rnd) == 0;
    }
};

// Unsigned 32-bit integer specialization
template <>
struct TypeMPFRFunctions<uint32_t>
{
    static int set(mpfr_ptr rop, uint32_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_ui(rop, op, rnd);
    }

    static uint32_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<uint32_t>(mpfr_get_ui(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_uint_p(op, rnd) == 0;
    }
};

// Signed 32-bit integer specialization
template <>
struct TypeMPFRFunctions<int32_t>
{
    static int set(mpfr_ptr rop, int32_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_si(rop, op, rnd);
    }

    static int32_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<int32_t>(mpfr_get_si(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_sint_p(op, rnd) == 0;
    }
};

// Unsigned 64-bit integer specialization
template <>
struct TypeMPFRFunctions<uint64_t>
{
    static int set(mpfr_ptr rop, uint64_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_ui(rop, op, rnd);
    }

    static uint64_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<uint64_t>(mpfr_get_ui(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_ulong_p(op, rnd) == 0;
    }
};

// Signed 64-bit integer specialization
template <>
struct TypeMPFRFunctions<int64_t>
{
    static int set(mpfr_ptr rop, int64_t op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_si(rop, op, rnd);
    }

    static int64_t get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return static_cast<int64_t>(mpfr_get_si(op, rnd));
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_fits_slong_p(op, rnd) == 0;
    }
};


// Specializations for floating point types
template <>
struct TypeMPFRFunctions<float>
{
    static int set(mpfr_ptr rop, float op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_flt(rop, op, rnd);
    }
    
    static float get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_get_flt(op, rnd);
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        double d = mpfr_get_d(op, rnd);
        return mpfr_cmp_d(op, d) == 0; 
    }
};

template <>
struct TypeMPFRFunctions<double>
{
    static int set(mpfr_ptr rop, double op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_set_d(rop, op, rnd);
    }

    static double get(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        return mpfr_get_d(op, rnd);
    }

    static bool fits(mpfr_srcptr op, mpfr_rnd_t rnd) noexcept
    {
        double d = mpfr_get_d(op, rnd);
        return mpfr_cmp_d(op, d) == 0; 
    }
};


 

template <typename T>
class FromMPF : public ConversionTrait
{
public:

    FromMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;

};

template <typename T>
bool FromMPF<T>::is_exact() const noexcept
{
    return false;
}

template <typename T>
void FromMPF<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    const auto* src_ptr = static_cast<mpfr_srcptr>(src);
    using funcs = TypeMPFRFunctions<T>;

    if (!is_exact() && exact) {
        RPY_CHECK(funcs::fits(src_ptr, MPFR_RNDN));
    }

    auto* dst_ptr = static_cast<T*>(dst);
    *dst_ptr = funcs::get(src_ptr, MPFR_RNDN);
    
}


template <>
class FromMPF<MPInt> : public ConversionTrait
{
public:
    FromMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;
    void unsafe_convert(void* dst, const void* src, bool exact) const override;

};

bool FromMPF<MPInt>::is_exact() const noexcept
{
    return false;
}

void FromMPF<MPInt>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    const auto* src_ptr = static_cast<const mpfr_srcptr>(src);
    auto* dst_ptr = static_cast<mpz_ptr>(dst);


    if (mpfr_get_z(dst_ptr, src_ptr, MPFR_RNDN) == 0) {
        RPY_THROW(std::runtime_error, "failed to convert to integer");
    }

    if (exact) {
        if (mpfr_cmp_z(src_ptr, dst_ptr) == 0) {
            RPY_THROW(std::runtime_error,
                "exact conversion requested but not delivered");
        }
    }

}


template <>
class FromMPF<MPRational> : public ConversionTrait
{
public:
    FromMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;
    void unsafe_convert(void* dst, const void* src, bool exact) const override;

};


bool FromMPF<MPRational>::is_exact() const noexcept
{
    return true;
}

void FromMPF<MPRational>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);
    ignore_unused(exact);

    auto* dst_ptr = static_cast<mpq_ptr>(dst);
    const auto* src_ptr = static_cast<mpfr_srcptr>(src);

    mpfr_get_q(dst_ptr, src_ptr);
}


template <typename T>
class ToMPF : public ConversionTrait
{
public:
    ToMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

template <typename T>
bool ToMPF<T>::is_exact() const noexcept
{
    const auto& tp = static_cast<const MPFloatType&>(*src_type());
    return tp.precision() >= std::numeric_limits<T>::digits;

}

template <typename T>
void ToMPF<T>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    const auto* src_ptr = static_cast<const T*>(src);
    using funcs = TypeMPFRFunctions<T>;

    auto* dst_ptr = static_cast<mpfr_ptr>(dst);
    funcs::set(dst_ptr, *src_ptr, MPFR_RNDN);

    if (!is_exact() && exact) {
        auto t = mpfr_flags_test(MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_OVERFLOW);
        RPY_CHECK(t == 0, "underflow or overflow occurred during conversion");
    }
    mpfr_flags_clear(MPFR_FLAGS_ALL);
}


template <>
class ToMPF<MPInt> : public ConversionTrait
{
public:

    ToMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;

    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

bool ToMPF<MPInt>::is_exact() const noexcept
{
    return true;
}

void ToMPF<MPInt>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    const auto* src_ptr = static_cast<mpz_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    if (mpfr_set_z(dst_ptr, src_ptr, MPFR_RNDN) == 0) {
        RPY_THROW(std::runtime_error, "failed to convert to integer");
    }

    if (exact) {
        auto t = mpfr_flags_test(MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_OVERFLOW);
        RPY_CHECK(t == 0, "underflow or overflow occurred");
        mpfr_flags_clear(MPFR_FLAGS_ALL);
    }
}


template <>
class ToMPF<MPRational> : public ConversionTrait
{
public:
    ToMPF(TypePtr from_type, TypePtr to_type)
        : ConversionTrait(std::move(from_type), std::move(to_type))
    {}

    RPY_NO_DISCARD bool is_exact() const noexcept override;
    void unsafe_convert(void* dst, const void* src, bool exact) const override;
};

bool ToMPF<MPRational>::is_exact() const noexcept
{
    return false;
}

void ToMPF<MPRational>::unsafe_convert(void* dst, const void* src, bool exact) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    const auto* src_ptr = static_cast<mpq_srcptr>(src);
    auto* dst_ptr = static_cast<mpfr_ptr>(dst);

    if (mpfr_set_q(dst_ptr, src_ptr, MPFR_RNDN) == 0) {
        RPY_THROW(std::runtime_error, "failed to convert to rational");
    }
    if (exact) {
        auto t = mpfr_flags_test(MPFR_FLAGS_UNDERFLOW | MPFR_FLAGS_OVERFLOW);
        RPY_CHECK(t == 0, "underflow or overflow occurred");
        mpfr_flags_clear(MPFR_FLAGS_ALL);
    }
}


template <typename T>
class FromMPFRFactory : public MPFloatConversionFromFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait> FromMPFRFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<MPFloat>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<T>);

    return std::make_unique<const FromMPF<T>>(std::move(from_type), std::move(to_type));
}

template <typename T>
class ToMPFRFactory : public MPFloatConversionToFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
std::unique_ptr<const ConversionTrait> ToMPFRFactory<T>::make(TypePtr from_type, TypePtr to_type) const
{
    RPY_DBG_ASSERT_EQ(from_type->id(), type_id_of<T>);
    RPY_DBG_ASSERT_EQ(to_type->id(), type_id_of<MPFloat>);

    return std::make_unique<const ToMPF<T>>(std::move(from_type), std::move(to_type));
}


}


#define ADD_FROM_FACTORY(Tp)                                                   \
    {hasher(type_id_of<Tp>), std::unique_ptr<FromMPFRFactory<Tp>>()}

const MPFloatConversionFromFactory* MPFloatConversionFromFactory::get_factory(
    const Type& type)
{
    using factory = std::unique_ptr<const MPFloatConversionFromFactory>;
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
        ADD_FROM_FACTORY(MPInt),
        ADD_FROM_FACTORY(MPRational)
   };

    if (auto it = cache.find(hasher(type.id())); it != cache.end()) {
        return it->second.get();
    }
    return nullptr;
}

#undef ADD_FROM_FACTORY

#define ADD_TO_FACTORY(Tp)                                                   \
    {hasher(type_id_of<Tp>), std::unique_ptr<ToMPFRFactory<Tp>>()}

const MPFloatConversionToFactory* MPFloatConversionToFactory::get_factory(
    const Type& type)
{
    using factory = std::unique_ptr<const MPFloatConversionToFactory>;
    constexpr Hash<string_view> hasher;
    static const boost::container::flat_map<hash_t, factory> cache {
        ADD_TO_FACTORY(double),
        ADD_TO_FACTORY(float),
        ADD_TO_FACTORY(int8_t),
        ADD_TO_FACTORY(int16_t),
        ADD_TO_FACTORY(int32_t),
        ADD_TO_FACTORY(uint8_t),
        ADD_TO_FACTORY(uint16_t),
        ADD_TO_FACTORY(uint32_t),
        ADD_TO_FACTORY(uint64_t),
        ADD_TO_FACTORY(MPInt),
        ADD_TO_FACTORY(MPRational)
   };

    if (auto it = cache.find(hasher(type.id())); it != cache.end()) {
        return it->second.get();
    }

    return nullptr;
}


#undef ADD_TO_FACTORY
