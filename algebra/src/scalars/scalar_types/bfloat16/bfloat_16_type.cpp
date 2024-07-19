//
// Created by sam on 24/06/24.
//

#include "bfloat_16_type.h"

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>

#include "bfloat16.h"


namespace rpy {
namespace scalars {
namespace implementations {
BFloat16Type::BFloat16Type()
    : Type("bf16",
           "BFloat16",
           {devices::TypeCode::BFloat, 2, 2, 1},
           devices::traits_of<BFloat16>())
{}

devices::Buffer
BFloat16Type::allocate(devices::Device device, dimn_t count) const
{
    return Type::allocate(device, count);
}
void* BFloat16Type::allocate_single() const { return Type::allocate_single(); }
void BFloat16Type::free_single(void* ptr) const { Type::free_single(ptr); }
bool BFloat16Type::supports_device(const devices::Device& device) const noexcept
{
    return Type::supports_device(device);
}
bool BFloat16Type::convertible_to(const Type& dest_type) const noexcept
{
    return Type::convertible_to(dest_type);
}
bool BFloat16Type::convertible_from(const Type& src_type) const noexcept
{
    return Type::convertible_from(src_type);
}
devices::TypeComparison BFloat16Type::compare_with(const Type& other
) const noexcept
{
    return Type::compare_with(other);
}
void BFloat16Type::copy(void* dst, const void* src, dimn_t count) const
{
    Type::copy(dst, src, count);
}
void BFloat16Type::move(void* dst, void* src, dimn_t count) const
{
    Type::move(dst, src, count);
}
void BFloat16Type::display(std::ostream& os, const void* ptr) const
{
    Type::display(os, ptr);
}
devices::ConstReference BFloat16Type::zero() const { return Type::zero(); }
devices::ConstReference BFloat16Type::one() const { return Type::one(); }
devices::ConstReference BFloat16Type::mone() const { return Type::mone(); }
}// namespace implementations
}// namespace scalars
}// namespace rpy
