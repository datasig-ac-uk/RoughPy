//
// Created by sam on 24/06/24.
//

#include "arbitrary_precision_rational_type.h"


#include <roughpy/core/smart_ptr.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>

namespace rpy {
namespace scalars {
namespace implementations {
ArbitraryPrecisionRationalType::ArbitraryPrecisionRationalType()
    : Type("rap",
           "Rational",
           {devices::TypeCode::ArbitraryPrecisionRational, 0, 0, 0},
           devices::traits_of<ArbitraryPrecisionRational>())
{}

devices::Buffer ArbitraryPrecisionRationalType::allocate(
        devices::Device device,
        dimn_t count
) const
{
    return Type::allocate(device, count);
}
void* ArbitraryPrecisionRationalType::allocate_single() const
{
    return new ArbitraryPrecisionRational();
}
void ArbitraryPrecisionRationalType::free_single(void* ptr) const
{
    delete static_cast<ArbitraryPrecisionRational*>(ptr);
}
bool ArbitraryPrecisionRationalType::supports_device(
        const devices::Device& device
) const noexcept
{
    return Type::supports_device(device);
}
devices::TypeComparison
ArbitraryPrecisionRationalType::compare_with(const Type& other) const noexcept
{
    return Type::compare_with(other);
}
void ArbitraryPrecisionRationalType::copy(
        void* dst,
        const void* src,
        dimn_t count
) const
{
    const auto* begin = static_cast<const ArbitraryPrecisionRational*>(dst);
    const auto* end = begin + count;
    ranges::copy(begin, end, static_cast<ArbitraryPrecisionRational*>(dst));
}
void ArbitraryPrecisionRationalType::move(void* dst, void* src, dimn_t count)
        const
{
    Type::move(dst, src, count);
}
void ArbitraryPrecisionRationalType::display(std::ostream& os, const void* ptr)
        const
{
    os << *static_cast<const ArbitraryPrecisionRational*>(ptr);
}
devices::ConstReference ArbitraryPrecisionRationalType::zero() const
{
    ArbitraryPrecisionRational zero{};
    return devices::ConstReference{&zero, this};
}
devices::ConstReference ArbitraryPrecisionRationalType::one() const
{
    ArbitraryPrecisionRational one{1};
    return devices::ConstReference{&one, this};
}
devices::ConstReference ArbitraryPrecisionRationalType::mone() const
{
    ArbitraryPrecisionRational mone{-1};
    return devices::ConstReference{&mone, this};
}

}// namespace implementations
}// namespace scalars
}// namespace rpy