//
// Created by sam on 27/11/24.
//

#include "polynomial_number.h"


#include "polynomial.h"


using namespace rpy;
using namespace rpy::generics;

bool PolynomialNumber::has_function(NumberFunction fn_id) const noexcept
{
    if (fn_id == NumberFunction::FromRational) { return true; }
    return false;
}

void PolynomialNumber::unsafe_abs(void* dst, const void* src) const noexcept
{
    // Do nothing, undefined
}

void PolynomialNumber::unsafe_from_rational(void* dst,
    int64_t numerator,
    int64_t denominator) const
{
    auto* dst_ptr = static_cast<Polynomial*>(dst);
    *dst_ptr = Polynomial(dtl::RationalCoeff{numerator, denominator});
}
