//
// Created by sam on 09/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"


namespace rpy::generics {


enum class TraitType
{
    Builtin,
    Dynamic
};



template <typename TraitImpl>
class Trait
{
    const TraitImpl* p_impl;

public:
    static constexpr TraitType Builtin = TraitType::Builtin;
    static constexpr TraitType Dynamic = TraitType::Dynamic;


    RPY_NO_DISCARD TraitType type() const noexcept
    {
        return p_impl->type();
    }
    RPY_NO_DISCARD string_view name() const noexcept
    {
        return p_impl->name();
    }

    template <typename... Args>
    RPY_NO_DISCARD decltype(auto) operator->(Args&&... args) const
        noexcept(p_impl->operator()(std::forward<Args>(args)...))
        -> decltype(p_impl->operator()(std::forward<Args>(args)...))
    {
        return p_impl->operator()(std::forward<Args>(args)...);
    }
};

}

#endif //ROUGHPY_GENERICS_TRAITS_TRAIT_H
