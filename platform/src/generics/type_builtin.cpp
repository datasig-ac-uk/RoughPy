//
// Created by sammorley on 16/11/24.
//



#include "roughpy/generics/type.h"

#include "builtin_types/double_type.h"
#include "builtin_types/float_type.h"
#include "builtin_types/signed_int_type.h"
#include "builtin_types/unsigned_int_type.h"

using namespace rpy;
using namespace rpy::generics;


const BuiltinTypes& rpy::generics::get_builtin_types() noexcept
{
    static BuiltinTypes builtins {
        FloatType::get(),
        DoubleType::get(),

        SignedInt8Type::get(),
        UnsignedInt8Type::get(),
        SignedInt16Type::get(),
        UnsignedInt16Type::get(),
        SignedInt32Type::get(),
        UnsignedInt32Type::get(),
        SignedInt64Type::get(),
        UnsignedInt64Type::get(),
    };

    return builtins;
}