//
// Created by sam on 13/03/23.
//

#ifndef ROUGHPY_RATIONALTYPE_H
#define ROUGHPY_RATIONALTYPE_H

#include "rational_type.h"
#include "standard_scalar_type.h"

namespace rpy {
namespace scalars {

class RationalType : public StandardScalarType<rational_scalar_type> {

public:

    RationalType();


};

}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_RATIONALTYPE_H
