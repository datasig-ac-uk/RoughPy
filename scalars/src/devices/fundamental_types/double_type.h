//
// Created by sam on 3/30/24.
//

#ifndef DOUBLE_TYPE_H
#define DOUBLE_TYPE_H

#include "devices/fundamental_type.h"

namespace rpy {
namespace devices {

class DoubleType : public FundamentalType<double> {

public:

    DoubleType();
};

extern const DoubleType double_type;

} // devices
} // rpy

#endif //DOUBLE_TYPE_H
