
#ifndef ROUGHPY_GENERICS_TRAITS_STATIC_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_STATIC_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "trait.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {




class ROUGHPY_PLATFORM_EXPORT StaticTrait : public Trait
{
public:
    static constexpr TraitType this_type = TraitType::Static;

    RPY_NO_DISCARD TraitType type() const noexcept final { return this_type; }

    RPY_NO_DISCARD virtual string_view id() const noexcept = 0;
};



}

#endif // ROUGHPY_GENERICS_TRAITS_STATIC_TRAIT_H