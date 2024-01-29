//
// Created by sam on 1/29/24.
//

#include "algebra.h"
#include "vector.h"

using namespace rpy;
using namespace rpy::algebra;

Algebra Algebra::multiply(const Vector& other) const { return Algebra(); }

Algebra& Algebra::multiply_inplace(const rpy::algebra::Vector& other)
{
    return *this;
}
