//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_EQUALITY_TRAIT_H
#define ROUGHPY_GENERICS_EQUALITY_TRAIT_H

#include "builtin_trait.h"
#include "roughpy/core/debug_assertion.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {

class ConstRef;

class ROUGHPY_PLATFORM_EXPORT EqualityTrait : public BuiltinTrait
{
protected:
    constexpr EqualityTrait() : BuiltinTrait(my_id) {}

    virtual ~EqualityTrait();

public:

    static constexpr BuiltinTraitID my_id = BuiltinTraitID::Equality;

    RPY_NO_DISCARD
    virtual bool
    unsafe_equals(const void* lhs, const void* rhs) const noexcept = 0;

    RPY_NO_DISCARD
    bool equals(ConstRef lhs, ConstRef rhs) const;
};



template <typename T>
class ROUGHPY_PLATFORM_NO_EXPORT EqualityTraitImpl : public EqualityTrait
{

public:

    EqualityTraitImpl() = default;

    RPY_NO_DISCARD
    bool unsafe_equals(const void* lhs, const void* rhs) const noexcept override
    {
        RPY_DBG_ASSERT_NE(lhs, nullptr);
        RPY_DBG_ASSERT_NE(rhs, nullptr);
        return *static_cast<const T*>(lhs) == *static_cast<const T*>(rhs);
    }

};

}

#endif //ROUGHPY_GENERICS_EQUALITY_TRAIT_H
