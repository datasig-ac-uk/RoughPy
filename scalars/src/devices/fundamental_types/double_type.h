//
// Created by sam on 3/30/24.
//

#ifndef DOUBLE_TYPE_H
#define DOUBLE_TYPE_H

#include "devices/fundamental_type.h"

namespace rpy {
namespace devices {

class RPY_LOCAL DoubleType : public FundamentalType<double> {

public:

    DoubleType();

    RPY_NO_DISCARD static const DoubleType* get() noexcept;
};


} // devices
} // rpy

#endif //DOUBLE_TYPE_H
