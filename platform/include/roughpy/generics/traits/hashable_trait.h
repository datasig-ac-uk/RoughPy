//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_HASHABLE_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_HASHABLE_TRAIT_H

#include "roughpy/core/hash.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "builtin_trait.h"

namespace rpy::generics {

class ConstReference;

class Hashable : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Hashable";

    RPY_NO_DISCARD string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return hashable; }

    RPY_NO_DISCARD virtual hash_t hash(ConstReference value) const noexcept = 0;
};


}


#endif //ROUGHPY_GENERICS_TRAITS_HASHABLE_TRAIT_H
