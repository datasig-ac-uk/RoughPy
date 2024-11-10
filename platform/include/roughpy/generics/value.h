//
// Created by sam on 09/11/24.
//

#ifndef ROUGHPY_GENERICS_VALUE_H
#define ROUGHPY_GENERICS_VALUE_H

#include "const_reference.h"
#include "reference.h"
#include "type.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {


class ConstrReference;
class Reference;


class ROUGHPY_PLATFORM_EXPORT Value
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
