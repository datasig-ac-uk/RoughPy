//
// Created by sam on 08/04/24.
//

#include "basis_key_type.h"

#include "basis_key.h"

#include <roughpy/scalars/devices/buffer.h>
#include <roughpy/scalars/devices/device_handle.h>

using namespace rpy;
using namespace rpy::algebra;
using namespace rpy::devices;

BasisKeyType::BasisKeyType()
    : Type("key", "BasisKey", basis_key_type_info, traits_of<BasisKey>())
{}



devices::Buffer
BasisKeyType::allocate(devices::Device device, dimn_t count) const
{
    return Type::allocate(device, count);
}
void* BasisKeyType::allocate_single() const { return Type::allocate_single(); }
void BasisKeyType::free_single(void* ptr) const { Type::free_single(ptr); }
bool BasisKeyType::supports_device(const devices::Device& device) const noexcept
{
    return Type::supports_device(device);
}
bool BasisKeyType::convertible_to(const devices::Type* dest_type) const noexcept
{
    return dest_type == this;
}
bool BasisKeyType::convertible_from(const devices::Type* src_type
) const noexcept
{
    return traits::is_integral(src_type) && !traits::is_signed(src_type);
}