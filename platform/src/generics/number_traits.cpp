//
// Created by sam on 09/11/24.
//

#include <stdexcept>

#include "roughpy/platform/errors.h"

#include "roughpy/generics/const_reference.h"
#include "roughpy/generics/traits.h"
// #include "roughpy/generics/type.h"
#include "roughpy/generics/value.h"



using namespace rpy;
using namespace rpy::generics;




Value Number::real(ConstReference value) const
{
    return Value(std::move(value));
}

Value Number::imaginary(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::minus(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::abs(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::sqrt(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::pow(ConstReference value, int exponent) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::exp(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::log(ConstReference value) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}

Value Number::from_rational(int64_t numerator, int64_t denominator) const
{
    RPY_THROW(std::runtime_error, "Not implemented");
}
