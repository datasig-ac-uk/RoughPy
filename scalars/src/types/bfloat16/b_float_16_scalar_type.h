//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_

#include "scalar_type.h"
#include "bfloat16_random_generator.h"
#include "scalar_implementations/bfloat.h"

namespace rpy {
namespace scalars {

class BFloat16ScalarType : public ScalarType
{
public:

    BFloat16ScalarType();


    static const ScalarType* get() noexcept;
};



}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_
