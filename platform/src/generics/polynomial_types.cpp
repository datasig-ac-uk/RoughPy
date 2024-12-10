//
// Created by sam on 27/11/24.
//


#include "roughpy/generics/type.h"

#include "polynomial_type/polynomial_type.h"


rpy::generics::TypePtr rpy::generics::get_polynomial_type() noexcept {
    return PolynomialType::get();
}