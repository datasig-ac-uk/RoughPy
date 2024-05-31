//
// Created by sam on 3/30/24.
//

#ifndef HALF_TYPE_H
#define HALF_TYPE_H

#include <roughpy/device_support/fundamental_type.h>
#include "scalar_implementations/half.h"

namespace rpy {
namespace devices {

class RPY_LOCAL HalfType : public Type {

public:
    HalfType();

    RPY_NO_DISCARD
    static const HalfType* get() noexcept;
};



} // devices
} // rpy

#endif //HALF_TYPE_H
