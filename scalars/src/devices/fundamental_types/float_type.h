//
// Created by sam on 3/30/24.
//

#ifndef FLOAT_TYPE_H
#define FLOAT_TYPE_H

#include "devices/fundamental_type.h"

namespace rpy {
namespace devices {

class RPY_LOCAL FloatType : public FundamentalType<float> {

public:
    FloatType();

    RPY_NO_DISCARD
    static const FloatType* get() noexcept;
};


} // devices
} // roy

#endif //FLOAT_TYPE_H
