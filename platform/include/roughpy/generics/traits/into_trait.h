//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_INTO_TRAIT_H
#define ROUGHPY_GENERICS_INTO_TRAIT_H

#include "roughpy/generics/type_ptr.h"

namespace rpy::generics {


class ConstReference;
class Reference;


class IntoTrait {
    TypePtr p_to;
public:
    virtual void unsafe_into_impl(void* dst, const void* src) const;

    void into(Reference dst, ConstReference src) const;
};



}


#endif //ROUGHPY_GENERICS_INTO_TRAIT_H
