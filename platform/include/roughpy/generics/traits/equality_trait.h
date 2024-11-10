//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_EQUALITY_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_EQUALITY_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "builtin_trait.h"


#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {

class ConstReference;

class ROUGHPY_PLATFORM_EXPORT Equality : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Equality";

    RPY_NO_DISCARD string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return equality; }

    RPY_NO_DISCARD virtual bool
    is_equal(ConstReference lhs, ConstReference rhs) const noexcept
            = 0;
    RPY_NO_DISCARD bool
    not_equal(ConstReference lhs, ConstReference rhs) const noexcept;
};

}

#endif //ROUGHPY_GENERICS_TRAITS_EQUALITY_TRAIT_H
