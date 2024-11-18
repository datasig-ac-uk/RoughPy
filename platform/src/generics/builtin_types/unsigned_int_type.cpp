//
// Created by sammorley on 18/11/24.
//

#include "unsigned_int_type.h"

#include "builtin_type_methods.h"

namespace rpy {
namespace generics {

template class BuiltinTypeBase<uint8_t>;
template class BuiltinTypeBase<uint16_t>;
template class BuiltinTypeBase<uint32_t>;
template class BuiltinTypeBase<uint64_t>;

} // generics
} // rpy

using namespace rpy;
using namespace rpy::generics;

string_view UnsignedInt8Type::name() const noexcept
{
    return "uint8";
}
const Type* UnsignedInt8Type::get() noexcept
{
    static const UnsignedInt8Type unsigned_int8;
    return &unsigned_int8;
}
string_view UnsignedInt16Type::name() const noexcept
{
    return "uint16";
}
const Type* UnsignedInt16Type::get() noexcept
{
    static const UnsignedInt16Type unsigned_int16;
    return &unsigned_int16;
}
string_view UnsignedInt32Type::name() const noexcept
{
    return "uint32";
}
const Type* UnsignedInt32Type::get() noexcept
{
    static const UnsignedInt32Type unsigned_int32;
    return &unsigned_int32;
}
string_view UnsignedInt64Type::name() const noexcept
{
    return "uint64";
}

const Type* UnsignedInt64Type::get() noexcept
{
    static const UnsignedInt64Type unsigned_int64;
    return &unsigned_int64;
}
