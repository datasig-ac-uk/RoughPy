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


void Type::move(void* dst, void* src, size_t count, bool uninit) const
{
    RPY_CHECK_NE(src, nullptr);
    copy_or_fill(dst, src, count, uninit);
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
