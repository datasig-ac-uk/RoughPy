//
// Created by sam on 16/02/24.
//

#include "free_tensor.h"

using namespace rpy;
using namespace rpy::algebra;

FreeTensor FreeTensor::exp() const
{
    return FreeTensor(basis(), scalar_type());
}
FreeTensor FreeTensor::log() const
{
    return FreeTensor(basis(), scalar_type());
}
FreeTensor FreeTensor::antipode() const
{
    return FreeTensor(basis(), scalar_type());
}

FreeTensor FreeTensor::fused_multiply_exp(const FreeTensor& other) const
{
    return FreeTensor(this->basis(), scalar_type());
}
FreeTensor& FreeTensor::fused_multiply_exp_inplcae(const FreeTensor& other)
{
    return *this;
}
