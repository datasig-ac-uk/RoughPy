//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_

#include "scalar_helpers/standard_scalar_type.h"

namespace rpy {
namespace scalars {

class DoubleType : public dtl::StandardScalarType<double>
{
    using base_t = dtl::ScalarContentType<double>;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_
