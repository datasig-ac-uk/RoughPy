//
// Created by sammorley on 16/11/24.
//



#include "roughpy/generics/type.h"

#include "builtin_types/double_type.h"
#include "builtin_types/float_type.h"

using namespace rpy;
using namespace rpy::generics;


const BuiltinTypes& rpy::generics::get_builtin_types() noexcept
{
    static BuiltinTypes builtins {
        FloatType::get(),
        DoubleType::get(),

        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };

    return builtins;
}