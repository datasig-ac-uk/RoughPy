//
// Created by user on 02/03/23.
//

#ifndef ROUGHPY_SCALARS_SRC_DOUBLE_TYPE_H
#define ROUGHPY_SCALARS_SRC_DOUBLE_TYPE_H

#include "standard_scalar_type.h"

namespace rpy {
namespace scalars {

class DoubleType : public StandardScalarType<double> {
public:
    DoubleType();
};

}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_SCALARS_SRC_DOUBLE_TYPE_H
