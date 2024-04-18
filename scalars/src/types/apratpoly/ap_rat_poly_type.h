//
// Created by sam on 3/30/24.
//

#ifndef AP_RAT_POLY_TYPE_H
#define AP_RAT_POLY_TYPE_H

#include "devices/fundamental_type.h"
#include "scalar_implementations/poly_rational.h"

namespace rpy {
namespace devices {

class RPY_LOCAL APRatPolyType : public FundamentalType<scalars::APPolyRat>
{
public:

    APRatPolyType();

    RPY_NO_DISCARD Buffer allocate(Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    bool supports_device(const Device& device) const noexcept override;

    RPY_NO_DISCARD
    static const APRatPolyType* get() noexcept;
};



} // devices
} // rpy

#endif //AP_RAT_POLY_TYPE_H
