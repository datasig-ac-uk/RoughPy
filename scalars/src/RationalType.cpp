//
// Created by sam on 13/03/23.
//

#include "RationalType.h"

using namespace rpy;
using namespace rpy::scalars;


RationalType::RationalType()
    : StandardScalarType<rational_scalar_type>("rational", "rational")
{}
