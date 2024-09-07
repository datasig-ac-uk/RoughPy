//
// Created by sam on 16/02/24.
//

#include "free_tensor.h"

using namespace rpy;
using namespace rpy::algebra;

FreeTensor FreeTensor::exp() const
{
    return FreeTensor();
}
FreeTensor FreeTensor::log() const
{
    return FreeTensor();
}
FreeTensor FreeTensor::antipode() const
{
    return FreeTensor();
}

FreeTensor FreeTensor::fused_multiply_exp(const FreeTensor& other) const
{
    return FreeTensor();
}
FreeTensor& FreeTensor::fused_multiply_exp_inplace(const FreeTensor& other)
{
    return *this;
}
