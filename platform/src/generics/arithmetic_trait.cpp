//
// Created by sam on 09/11/24.
//


#include "roughpy/generics/const_reference.h"
#include "roughpy/generics/reference.h"
#include "roughpy/generics/traits.h"
#include "roughpy/generics/value.h"


using namespace rpy;
using namespace rpy::generics;


Value Arithmetic::add(ConstReference lhs, ConstReference rhs) const
{
    Value result(lhs);
    this->add_inplace(result, rhs);
    return result;
}


Value Arithmetic::sub(ConstReference lhs, ConstReference rhs) const
{
    Value result(lhs);
    this->sub_inplace(result, rhs);
    return result;
}


Value Arithmetic::mul(ConstReference lhs, ConstReference rhs) const
{
    Value result(lhs);
    this->mul_inplace(result, rhs);
    return result;
}

Value Arithmetic::div(ConstReference lhs, ConstReference rhs) const
{
    Value result(lhs);
    this->div_inplace(result, rhs);
    return result;
}


