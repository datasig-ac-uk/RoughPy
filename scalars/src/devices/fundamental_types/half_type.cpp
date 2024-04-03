//
// Created by sam on 3/30/24.
//

#include "half_type.h"

using namespace rpy;
using namespace rpy::devices;

HalfType::HalfType() : FundamentalType("f16", "HPReal"){  }


const HalfType devices::half_type;
