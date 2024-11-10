//
// Created by sam on 09/11/24.
//


#include "roughpy/generics/traits/ordering_trait.h"
#include "roughpy/generics/const_reference.h"



using namespace rpy;
using namespace rpy::generics;


bool Ordering::comparable(ConstReference lhs, ConstReference rhs) const noexcept
{
    return this->compare(std::move(lhs), std::move(rhs)) != Incomparable;
}

bool Ordering::less(ConstReference lhs, ConstReference rhs) const noexcept
{
    return this->compare(std::move(lhs), std::move(rhs)) == LessThan;
}

bool Ordering::less_equal(ConstReference lhs, ConstReference rhs) const noexcept
{
    auto result = this->compare(std::move(lhs), std::move(rhs));
    return result == LessThan || result == Equal;
}

bool Ordering::greater(ConstReference lhs, ConstReference rhs) const noexcept
{
    return this->compare(std::move(lhs), std::move(rhs)) == GreaterThan;
}

bool Ordering::greater_equal(ConstReference lhs, ConstReference rhs) const noexcept
{
    auto result = this->compare(std::move(lhs), std::move(rhs));
    return result == GreaterThan || result == Equal;
}

