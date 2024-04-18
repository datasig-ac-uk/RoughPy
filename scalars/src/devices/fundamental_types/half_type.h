//
// Created by sam on 3/30/24.
//

#ifndef HALF_TYPE_H
#define HALF_TYPE_H

#include "devices/fundamental_type.h"
#include "scalar_implementations/half.h"

namespace rpy {
namespace devices {

class RPY_LOCAL HalfType : public FundamentalType<scalars::Half> {

public:
    HalfType();

    RPY_NO_DISCARD
    static const HalfType* get() noexcept;
};



} // devices
} // rpy

#endif //HALF_TYPE_H
