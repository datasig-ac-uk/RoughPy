//
// Created by sam on 3/30/24.
//

#include "double_type.h"


using namespace rpy;
using namespace rpy::devices;

DoubleType::DoubleType() : FundamentalType("f64", "DPReal") {}

const DoubleType devices::double_type;
