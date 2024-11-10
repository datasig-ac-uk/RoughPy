
#ifndef ROUGHPY_GENERICS_TRAITS_DYNAMIC_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_DYNAMIC_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "trait.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {


class ROUGHPY_PLATFORM_EXPORT DynamicTrait : public Trait
{

public:
    static constexpr TraitType this_type = TraitType::Dynamic;

    RPY_NO_DISCARD TraitType type() const noexcept final { return this_type; }

    RPY_NO_DISCARD virtual string_view id() const noexcept = 0;
};


}


#endif // ROUGHPY_GENERICS_TRAITS_DYNAMIC_TRAIT_H