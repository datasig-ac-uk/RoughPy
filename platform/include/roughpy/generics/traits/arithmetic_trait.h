//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TRAITS_ARITHMETIC_TRAIT_H
#define ROUGHPY_GENERICS_TRAITS_ARITHMETIC_TRAIT_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "builtin_trait.h"


namespace rpy::generics {

class ConstReference;
class Reference;
class Value;

class Arithmetic : public BuiltinTrait
{
public:
    static constexpr string_view this_name = "Arithmetic";

    RPY_NO_DISCARD string_view name() const noexcept final { return this_name; }

    RPY_NO_DISCARD size_t index() const noexcept final { return arithmetic; }

    virtual void add_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void sub_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void mul_inplace(Reference lhs, ConstReference rhs) const = 0;
    virtual void div_inplace(Reference lhs, ConstReference rhs) const = 0;

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
