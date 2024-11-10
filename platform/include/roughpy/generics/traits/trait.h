//
// Created by sam on 09/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy_platform_export.h"

namespace rpy::generics {


enum class TraitType
{
    Builtin,
    Static,
    Dynamic
};

class ROUGHPY_PLATFORM_EXPORT Trait
{
public:
    static constexpr TraitType Builtin = TraitType::Builtin;
    static constexpr TraitType Static = TraitType::Static;
    static constexpr TraitType Dynamic = TraitType::Dynamic;

    virtual ~Trait() = default;

    RPY_NO_DISCARD virtual TraitType type() const noexcept = 0;

    RPY_NO_DISCARD virtual string_view name() const noexcept = 0;
};

}

#endif //ROUGHPY_GENERICS_TRAITS_TRAIT_H
