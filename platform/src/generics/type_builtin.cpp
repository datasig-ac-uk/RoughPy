//
// Created by sammorley on 16/11/24.
//



#include "roughpy/generics/type.h"


using namespace rpy;
using namespace rpy::generics;


const BuiltinTypes& rpy::generics::get_builtin_types() noexcept
{
    static BuiltinTypes builtins {
        nullptr,
        nullptr,

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