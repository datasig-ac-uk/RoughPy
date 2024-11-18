//
// Created by sammorley on 18/11/24.
//

#include "float_type.h"

#include "builtin_type_ids.h"
#include "builtin_type_methods.h"


namespace rpy::generics {

template class BuiltinTypeBase<float>;

}

using namespace rpy;
using namespace rpy::generics;
string_view FloatType::name() const noexcept
{
    return "float";
}
string_view FloatType::id() const noexcept
{
    return type_id_of<float>;
}

const Type* FloatType::get() noexcept
{
    static const FloatType object;
    return &object;
}

