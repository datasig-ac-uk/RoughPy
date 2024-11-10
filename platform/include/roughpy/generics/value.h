//
// Created by sam on 09/11/24.
//

#ifndef ROUGHPY_GENERICS_VALUE_H
#define ROUGHPY_GENERICS_VALUE_H

#include "const_reference.h"
#include "reference.h"


namespace rpy::generics {

class Type;

using TypePtr = const Type*;

class ConstrReference;
class Reference;


class Value
{
    TypePtr p_type;

public:

    explicit Value(ConstReference ref) {}


    operator ConstReference() const noexcept
    {
        return {};
    }

    operator Reference() noexcept
    {
        return {};
    }

};

}

#endif //ROUGHPY_GENERICS_VALUE_H
