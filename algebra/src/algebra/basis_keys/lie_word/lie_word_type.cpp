//
// Created by sam on 8/13/24.
//

#include "lie_word_type.h"

#include <roughpy/devices/device_handle.h>

#include "lie_word.h"

using namespace rpy;
using namespace rpy::algebra;

LieWordType::LieWordType()
    : Type("lie_word",
           "LieWord",
           {devices::TypeCode::KeyType, sizeof(LieWord), alignof(LieWord), 1},
           devices::traits_of<LieWord>())
{}

void* LieWordType::allocate_single() const { return new LieWord{}; }
void LieWordType::free_single(void* ptr) const
{
    delete static_cast<LieWord*>(ptr);
}
bool LieWordType::supports_device(const devices::Device& device) const noexcept
{
    return device->is_host();
}
bool LieWordType::convertible_to(const Type& dest_type) const noexcept
{
    return Type::convertible_to(dest_type);
}
bool LieWordType::convertible_from(const Type& src_type) const noexcept
{
    return Type::convertible_from(src_type);
}
devices::TypeComparison LieWordType::compare_with(const Type& other
) const noexcept
{
    return Type::compare_with(other);
}
void LieWordType::copy(void* dst, const void* src, dimn_t count) const
{
    Type::copy(dst, src, count);
}
void LieWordType::move(void* dst, void* src, dimn_t count) const
{
    Type::move(dst, src, count);
}
void LieWordType::display(std::ostream& os, const void* ptr) const
{
    Type::display(os, ptr);
}
