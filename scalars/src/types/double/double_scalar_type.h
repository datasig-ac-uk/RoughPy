//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_

#include "scalar_type.h"

namespace rpy {
namespace scalars {

class DoubleType : public ScalarType
{
public:

    DoubleType();

    static const ScalarType* get() noexcept;

};




}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_DOUBLE_DOUBLE_TYPE_H_
