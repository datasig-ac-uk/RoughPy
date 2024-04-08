//
// Created by sam on 3/30/24.
//

#ifndef AP_RATIONAL_TYPE_H
#define AP_RATIONAL_TYPE_H

#include "devices/fundamental_type.h"
#include "scalar_implementations/arbitrary_precision_rational.h"

namespace rpy {
namespace devices {

class RPY_LOCAL APRationalType : public FundamentalType<scalars::ArbitraryPrecisionRational> {

public:

    APRationalType();

    RPY_NO_DISCARD Buffer allocate(Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    bool supports_device(const Device& device) const noexcept override;
};

extern const APRationalType arbitrary_precision_rational_type;

}
}

#endif //AP_RATIONAL_TYPE_H
