//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_FROM_TRAIT_H
#define ROUGHPY_GENERICS_FROM_TRAIT_H


#include "roughpy/generics/type_ptr.h"

#include "builtin_trait.h"

namespace rpy::generics {

class Reference;
class ConstReference;
class Value;

class FromTrait
{
    TypePtr p_from;
    TypePtr p_to;
public:

    FromTrait(TypePtr from, TypePtr to)
        : p_from(std::move(from)),
          p_to(std::move(to))
    {}

    virtual ~FromTrait();
    virtual void unsafe_from_impl(void* dst, const void* src) const noexcept = 0;

    void from(Reference dst, ConstReference src) const;

    RPY_NO_DISCARD
    Value from(ConstReference src) const;

};



}

#endif //ROUGHPY_GENERICS_FROM_TRAIT_H
