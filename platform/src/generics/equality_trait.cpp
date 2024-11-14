//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/equality_trait.h"

#include "roughpy/generics/values.h"

using namespace rpy;
using namespace rpy::generics;

EqualityTrait::~EqualityTrait() = default;

bool EqualityTrait::equals(ConstRef lhs, ConstRef rhs) const
{
    if (!lhs.is_valid() && !rhs.is_valid()) {
        return true;
    }

    if (lhs.is_valid() || rhs.is_valid()) {
        return false;
    }

    RPY_CHECK_EQ(lhs.type(), rhs.type());
    return unsafe_equals(lhs.data(), rhs.data());
}
