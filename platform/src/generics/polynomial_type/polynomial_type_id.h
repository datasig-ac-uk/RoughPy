//
// Created by sammorley on 02/12/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_TYPE_ID_H
#define ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_TYPE_ID_H


#include "roughpy/generics/type_ptr.h"

namespace rpy::generics {

class Polynomial;

template <>
inline constexpr string_view type_id_of<Polynomial> = "poly";


}

#endif //ROUGHPY_GENERICS_INTERNAL_POLYNOMIAL_TYPE_ID_H
