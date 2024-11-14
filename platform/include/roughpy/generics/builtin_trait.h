//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_BUILTIN_TRAIT_H
#define ROUGHPY_GENERICS_BUILTIN_TRAIT_H

#include "roughpy/core/check.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {


class EqualityTrait;
class HashTrait;
class OrderingTrait;
class ArithmeticTrait;
class NumberTrait;

enum class BuiltinTraitID : size_t
{
    Equality = 0,
    Hash,
    Ordering,
    Arithmetic,
    Number
};


class ROUGHPY_PLATFORM_EXPORT BuiltinTrait {
    BuiltinTraitID m_id;

protected:

    explicit constexpr BuiltinTrait(BuiltinTraitID id) : m_id(id) {}
public:

    RPY_NO_DISCARD
    constexpr BuiltinTraitID id() const noexcept { return m_id; }

};


template <typename TraitObject>
constexpr enable_if_t<is_base_of_v<BuiltinTrait, TraitObject>, const TraitObject*>
trait_cast(const BuiltinTrait* trait) noexcept
{
    if (trait == nullptr || trait->id() == TraitObject::my_id) {
        return nullptr;
    }
    return static_cast<const TraitObject*>(trait);
}


}

#endif //ROUGHPY_GENERICS_BUILTIN_TRAIT_H
