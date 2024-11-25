//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/type.h"

#include <memory>

#include <roughpy/core/macros.h>

#include "roughpy/generics/conversion_trait.h"

using namespace rpy;
using namespace rpy::generics;

bool Type::parse_from_string(void* data, string_view str) const noexcept
{
    return false;
}
std::unique_ptr<const ConversionTrait> Type::convert_to(const Type& type
) const noexcept
{
    return type.convert_from(*this);
}
std::unique_ptr<const ConversionTrait> Type::convert_from(const Type& type
) const noexcept
{
    return nullptr;
}

const BuiltinTrait* Type::get_builtin_trait(BuiltinTraitID id) const noexcept
{
    return nullptr;
}
// const Trait* Type::get_trait(string_view id) const noexcept { return nullptr;
// }
hash_t Type::hash_of(const void* value) const noexcept
{
    return 0;
}
