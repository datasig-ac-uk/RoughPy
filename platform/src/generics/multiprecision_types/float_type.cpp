//
// Created by sammorley on 25/11/24.
//

#include "float_type.h"


#include <gmp.h>
#include <mpfr.h>

using namespace rpy;
using namespace rpy::generics;

MPFloatType::MPFloatType(size_t precision) {}

const std::type_info& MPFloatType::type_info() const noexcept {}

BasicProperties MPFloatType::basic_properties() const noexcept {}

size_t MPFloatType::object_size() const noexcept {}

void* MPFloatType::allocate_object() const {}

void MPFloatType::free_object(void* ptr) const {}

bool MPFloatType::parse_from_string(void* data, string_view str) const noexcept
{
    return RefCountedMiddle<Type>::parse_from_string(data, str);
}

void MPFloatType::copy_or_move(void* dst,
    const void* src,
    size_t count,
    bool move) const {}

void MPFloatType::destroy_range(void* data, size_t count) const {}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_to(const Type& type) const noexcept
{
    return RefCountedMiddle<Type>::convert_to(type);
}

std::unique_ptr<const ConversionTrait> MPFloatType::
convert_from(const Type& type) const noexcept
{
    return RefCountedMiddle<Type>::convert_from(type);
}

const BuiltinTrait* MPFloatType::
get_builtin_trait(BuiltinTraitID id) const noexcept
{
    return RefCountedMiddle<Type>::get_builtin_trait(id);
}

const std::ostream& MPFloatType::display(std::ostream& os,
    const void* value) const {}

hash_t MPFloatType::hash_of(const void* value) const noexcept
{
    return RefCountedMiddle<Type>::hash_of(value);
}
