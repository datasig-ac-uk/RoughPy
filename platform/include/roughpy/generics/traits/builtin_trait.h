

#ifndef ROUGHPY_GENERICS_TRAITS_BUILTIN_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_BUILTIN_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "trait.h"

namespace rpy::generics {


enum class BuiltinTraits : size_t
{
    Hashable = 0,
    Equality,
    Arithmetic,
    Ordering,
    Number
};

inline constexpr size_t builtin_trait_count = 5;

class BuiltinTrait : public Trait
{
public:
    static constexpr size_t hashable
            = static_cast<size_t>(BuiltinTraits::Hashable);
    static constexpr size_t equality
            = static_cast<size_t>(BuiltinTraits::Equality);
    static constexpr size_t arithmetic
            = static_cast<size_t>(BuiltinTraits::Arithmetic);
    static constexpr size_t ordering
            = static_cast<size_t>(BuiltinTraits::Ordering);
    static constexpr size_t number = static_cast<size_t>(BuiltinTraits::Number);

    static constexpr TraitType this_type = TraitType::Builtin;

    RPY_NO_DISCARD TraitType type() const noexcept final { return this_type; }

    RPY_NO_DISCARD virtual size_t index() const noexcept = 0;
};




}


#endif// ROUGHPY_GENERICS_TRAITS_BUILTIN_TRAIT_H
