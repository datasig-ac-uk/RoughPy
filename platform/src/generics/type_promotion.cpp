//
// Created by sammorley on 16/11/24.
//


#include "roughpy/generics/type.h"



using namespace rpy;
using namespace rpy::generics;


TypePtr generics::compute_promotion(const Type* lhs, const Type* rhs) noexcept
{
    if (lhs == nullptr) {
        return rhs;
    }
    if (rhs == nullptr) {
        return lhs;
    }

    // TODO: Create a mechanism for determining which type should be taken

    return nullptr;
}
