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
    if (RPY_LIKELY(lhs == rhs)) {
        return lhs;
    }

    /*
     * The algorithm is as follows:
     *   1) if lhs is exactly convertible to rhs then return rhs
     *   2) if rhs is exactly convertible to lhs then return lhs
     *
     * In the future, we might wish to extend this algorithm to allow for a non-
     * exact conversion or promote to a higher type beyond the two that are
     * given.
     *
     * We specifically use "convertible to" because this will capture both
     * internally defined conversion and "from conversion" from the other type
     */
    if (const auto l2r_conv = lhs->convert_to(*rhs);
        l2r_conv && l2r_conv->is_exact()) {
        return rhs;
    }

    if (const auto r2l_conv = rhs->convert_to(*lhs);
        r2l_conv && r2l_conv->is_exact()) {
        return lhs;
    }

    return nullptr;
}
