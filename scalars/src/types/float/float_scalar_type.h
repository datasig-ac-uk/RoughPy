//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_FLOAT_FLOAT_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_FLOAT_FLOAT_TYPE_H_

#include "scalar_helpers/standard_scalar_type.h"

namespace rpy {
namespace scalars {

class FloatType : public ScalarType
{
public:

    FloatType();


    static const ScalarType* get() noexcept;
};


RPY_LOCAL extern const FloatType float_type;

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_FLOAT_FLOAT_TYPE_H_
