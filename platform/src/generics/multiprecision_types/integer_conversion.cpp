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
    static int set(mpz_ptr rop, T op) noexcept;

    static T get(mpz_srcptr op) noexcept;

    static bool compare(mpz_srcptr lhs, T rhs) noexcept;
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
class FromMPIntFactory : public MPIntegerConversionFromFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait> make(TypePtr from_type,
        TypePtr to_type) const override;
};

template <typename T>
class ToMPIntFactory : public MPIntegerConversionToFactory
{
public:
    RPY_NO_DISCARD std::unique_ptr<const ConversionTrait>
    make(TypePtr from_type, TypePtr to_type) const override;
};



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