//
// Created by sam on 3/30/24.
//

#include "ap_rational_type.h"

#include <roughpy/core/alloc.h>

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>

using namespace rpy;
using namespace rpy::devices;

rpy::devices::APRationalType::APRationalType()
    : Type("Rational",
           "Rational",
           {TypeCode::ArbitraryPrecisionRational,
            sizeof(scalars::ArbitraryPrecisionRational),
            alignof(scalars::ArbitraryPrecisionRational),
            1},
           traits_of<scalars::ArbitraryPrecisionRational>())
{}

Buffer APRationalType::allocate(Device device, dimn_t count) const
{
    RPY_CHECK(device->is_host());
    return Type::allocate(device, count);
}
void* APRationalType::allocate_single() const
{
    auto* ptr = Type::allocate_single();
    construct_inplace(static_cast<scalars::ArbitraryPrecisionRational*>(ptr));
    return ptr;
}
void APRationalType::free_single(void* ptr) const
{
    std::destroy_at(static_cast<scalars::ArbitraryPrecisionRational*>(ptr));
    Type::free_single(ptr);
}
bool APRationalType::supports_device(const Device& device) const noexcept
{
    return device->is_host();
}
const APRationalType* APRationalType::get() noexcept
{
    static const APRationalType type;
    return &type;
}
