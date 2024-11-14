//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_HASH_TRAIT_H
#define ROUGHPY_GENERICS_HASH_TRAIT_H

#include "builtin_trait.h"

#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"
#include "roughpy/core/hash.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {

class ConstRef;

class ROUGHPY_PLATFORM_EXPORT HashTrait : public BuiltinTrait
{
protected:

    constexpr HashTrait() : BuiltinTrait(my_id) {}

public:

    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Hash;

    virtual ~HashTrait();

    RPY_NO_DISCARD
    virtual hash_t unsafe_hash(const void* value) const noexcept = 0;

    RPY_NO_DISCARD
    hash_t hash(ConstRef value) const;

};



template <typename T>
class ROUGHPY_PLATFORM_NO_EXPORT HashTraitImpl : public HashTrait
{
public:

    RPY_NO_DISCARD
    hash_t unsafe_hash(const void* value) const noexcept override;
};

template <typename T>
hash_t HashTraitImpl<T>::unsafe_hash(const void* value) const noexcept
{
    return hash_value(*static_cast<const T*>(value));
}


}

#endif //ROUGHPY_GENERICS_HASH_TRAIT_H
