//
// Created by sam on 3/30/24.
//

#ifndef B_FLOAT_16_TYPE_H
#define B_FLOAT_16_TYPE_H

#include "devices/fundamental_type.h"
#include "scalar_implementations/bfloat.h"

namespace rpy {
namespace devices {

class BFloat16Type : public FundamentalType<scalars::BFloat16> {

public:

    BFloat16Type();


};


extern const BFloat16Type bfloat16_type;

} // devices
} // rpy

#endif //B_FLOAT_16_TYPE_H
