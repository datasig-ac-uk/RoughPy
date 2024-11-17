//
// Created by sam on 16/11/24.
//

#include "double_type.h"


#include "builtin_type_methods.h"

namespace rpy::generics {
template class BuiltinTypeBase<double>;
}


using namespace rpy;
using namespace rpy::generics;

string_view DoubleType::name() const noexcept
{
    return "double";
}

string_view DoubleType::id() const noexcept { return type_id_of<double>; }

const Type* DoubleType::get() noexcept
{
    static const DoubleType object;
    return &object;
}
