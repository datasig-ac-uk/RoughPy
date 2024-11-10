//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_TRAIT_CASTS_H
#define ROUGHPY_GENERICS_TRAITS_TRAIT_CASTS_H

#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/errors.h"

#include "builtin_trait.h"
#include "dynamic_trait.h"
#include "static_trait.h"
#include "trait.h"

namespace rpy::generics {


template <typename T>
inline constexpr bool is_trait_base_v = is_same_v<T, BuiltinTrait>
        || is_same_v<T, StaticTrait> || is_same_v<T, DynamicTrait>;

template <typename T>
enable_if_t<is_trait_base_v<T> && !is_const_v<T>, add_lvalue_reference_t<T>>
trait_cast(Trait& trait)
{
    if (trait.type() != T::this_type) { RPY_THROW(std::bad_cast); }
    return static_cast<T&>(trait);
}

template <typename T>
enable_if_t<is_trait_base_v<T>, add_lvalue_reference_t<add_const_t<T>>>
trait_cast(const Trait& trait)
{
    if (trait.type() != T::this_type) { RPY_THROW(std::bad_cast); }
    return static_cast<const T&>(trait);
}



}


#endif //ROUGHPY_GENERICS_TRAITS_TRAIT_CASTS_H
