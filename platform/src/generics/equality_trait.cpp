//
// Created by sam on 09/11/24.
//

#include "roughpy/generics/const_reference.h"
#include "roughpy/generics/traits.h"


using namespace rpy;
using namespace rpy::generics;


bool Equality::not_equal(ConstReference lhs, ConstReference rhs) const noexcept {
    return !this->is_equal(std::move(lhs), std::move(rhs));
}
