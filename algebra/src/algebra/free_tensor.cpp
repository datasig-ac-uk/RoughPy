//
// Created by sam on 16/02/24.
//

#include "free_tensor.h"

using namespace rpy;
using namespace rpy::algebra;

FreeTensor FreeTensor::new_like(const FreeTensor& arg) noexcept
{
    return {};
}
FreeTensor FreeTensor::clone(const FreeTensor& arg) noexcept
{
    return {};
}
FreeTensor FreeTensor::from(Vector&& arg) noexcept
{
    return {};
}
FreeTensor::FreeTensor()
    : AlgebraBase(Vector())
{}
FreeTensor&
FreeTensor::fma(const Vector& lhs, const Vector& rhs, const ops::Operator& op)
{
    return *this;
}
FreeTensor& FreeTensor::multiply_inplace(
        const Vector& lhs,
        const Vector& rhs,
        const ops::Operator& op
)
{
    return *this;
}

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
