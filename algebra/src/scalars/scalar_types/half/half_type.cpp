//
// Created by sam on 24/06/24.
//

#include "half_type.h"

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/type.h>
#include <roughpy/devices/value.h>

#include <roughpy/platform/alloc.h>

#include "half.h"

namespace rpy {
namespace scalars {
namespace implementations {

HalfType::HalfType()
    : Type("f16",
           "HPReal",
           {devices::TypeCode::Float, 2, 2, 1},
           devices::traits_of<Half>())
{}
devices::Buffer HalfType::allocate(devices::Device device, dimn_t count) const
{
    return Type::allocate(device, count);
}
void* HalfType::allocate_single() const
{
    return nullptr;
}
void HalfType::free_single(void* ptr) const { }
bool HalfType::supports_device(const devices::Device& device) const noexcept
{
    return Type::supports_device(device);
}
bool HalfType::convertible_to(const Type& dest_type) const noexcept
{
    return Type::convertible_to(dest_type);
}
bool HalfType::convertible_from(const Type& src_type) const noexcept
{
    return Type::convertible_from(src_type);
}
devices::TypeComparison HalfType::compare_with(const Type& other) const noexcept
{
    return Type::compare_with(other);
}
void HalfType::copy(void* dst, const void* src, dimn_t count) const
{
    Type::copy(dst, src, count);
}
void HalfType::move(void* dst, void* src, dimn_t count) const
{
    Type::move(dst, src, count);
}
void HalfType::display(std::ostream& os, const void* ptr) const
{
    Type::display(os, ptr);
}
devices::ConstReference HalfType::zero() const { return Type::zero(); }
devices::ConstReference HalfType::one() const { return Type::one(); }
devices::ConstReference HalfType::mone() const { return Type::mone(); }

}// namespace implementations
}// namespace scalars
}// namespace rpy