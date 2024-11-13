//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_ARITHMETIC_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_ARITHMETIC_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "builtin_trait.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {

class ConstReference;
class Reference;
class Value;

class ROUGHPY_PLATFORM_EXPORT Arithmetic : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Arithmetic";

    RPY_NO_DISCARD string_view name() const noexcept { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return arithmetic; }


    virtual void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept = 0;
    virtual void unsafe_sub_inplace(void* lhs, const void* rhs) const noexcept = 0;
    virtual void unsafe_mul_inplace(void* lhs, const void* rhs) const noexcept = 0;
    virtual void unsafe_div_inplace(void* lhs, const void* rhs) const noexcept = 0;


    void add_inplace(Reference lhs, ConstReference rhs) const;
    void sub_inplace(Reference lhs, ConstReference rhs) const;
    void mul_inplace(Reference lhs, ConstReference rhs) const;
    void div_inplace(Reference lhs, ConstReference rhs) const;

    RPY_NO_DISCARD virtual Value
    add(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD virtual Value
    sub(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD virtual Value
    mul(ConstReference lhs, ConstReference rhs) const;
    RPY_NO_DISCARD virtual Value
    div(ConstReference lhs, ConstReference rhs) const;
};

}

#endif //ROUGHPY_GENERICS_TRAITS_ARITHMETIC_TRAIT_H
