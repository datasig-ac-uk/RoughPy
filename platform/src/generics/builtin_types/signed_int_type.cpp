//
// Created by sammorley on 18/11/24.
//

#include "signed_int_type.h"

#include "builtin_type_methods.h"

namespace rpy:: generics {

template class BuiltinTypeBase<int8_t>;
template class BuiltinTypeBase<int16_t>;
template class BuiltinTypeBase<int32_t>;
template class BuiltinTypeBase<int64_t>;

} // rpy::generics

using namespace rpy;
using namespace rpy::generics;

string_view SignedInt8Type::name() const noexcept
{
    return "int8";
}
const Type* SignedInt8Type::get() noexcept
{
    static const SignedInt8Type signed_int8;
    return &signed_int8;
}
string_view SignedInt16Type::name() const noexcept
{
    return "int16";
}
const Type* SignedInt16Type::get() noexcept
{
    static const SignedInt16Type signed_int16;
    return &signed_int16;
}
string_view SignedInt32Type::name() const noexcept
{
    return "int32";
}
const Type* SignedInt32Type::get() noexcept
{
    static const SignedInt32Type signed_int32;
    return &signed_int32;
}
string_view SignedInt64Type::name() const noexcept
{
    return "int64";
}
const Type* SignedInt64Type::get() noexcept
{
    static const SignedInt64Type signed_int64;
    return &signed_int64;
}
