//
// Created by sam on 3/30/24.
//

#ifndef B_FLOAT_16_TYPE_H
#define B_FLOAT_16_TYPE_H

#include "scalar_implementations/bfloat.h"
#include "devices/fundamental_type.h"

namespace rpy {
namespace devices {

class RPY_LOCAL BFloat16Type : public FundamentalType<scalars::BFloat16> {

public:

    BFloat16Type();

    RPY_NO_DISCARD
    static const BFloat16Type* get() noexcept;
};



} // devices
} // rpy

#endif //B_FLOAT_16_TYPE_H
