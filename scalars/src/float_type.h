//
// Created by user on 02/03/23.
//

#ifndef ROUGHPY_SCALARS_SRC_FLOAT_TYPE_H
#define ROUGHPY_SCALARS_SRC_FLOAT_TYPE_H

#include "standard_scalar_type.h"

namespace rpy {
namespace scalars {

class FloatType : public StandardScalarType<float>{
public:
    FloatType();
};

}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_SCALARS_SRC_FLOAT_TYPE_H
