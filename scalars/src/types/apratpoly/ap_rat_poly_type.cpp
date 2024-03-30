//
// Created by sam on 3/30/24.
//

#include "ap_rat_poly_type.h"

using namespace rpy;
using namespace rpy::devices;

APRatPolyType::APRatPolyType()
    : FundamentalType("APRatPoly", "APRatPoly")
{}

Buffer APRatPolyType::allocate(Device device, dimn_t count) const
{
    RPY_CHECK(device->is_host());
    auto buf = Type::allocate(device, count);


    return buf;
}
void* APRatPolyType::allocate_single() const
{
    auto* ptr = Type::allocate_single();
    construct_inplace(static_cast<APRatPolyType*>(ptr));
    return ptr;
}
void APRatPolyType::free_single(void* ptr) const
{
    std::destroy_at(static_cast<APRatPolyType*>(ptr));
    Type::free_single(ptr);
}
bool APRatPolyType::supports_device(const Device& device) const noexcept
{
    return device->is_host();
}





const APRatPolyType devices::arbitrary_precision_rational_poly_type;
