//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_

#include "scalar_helpers/standard_scalar_type.h"
#include "scalar_types.h"
#include "bfloat16_random_generator.h"

namespace rpy {
namespace scalars {

class BFloat16Type : public dtl::StandardScalarType<bfloat16>
{
    using base_t = dtl::StandardScalarType<bfloat16>;


public:

    BFloat16Type();


    static const ScalarType* get() noexcept;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_BFLOAT16_B_FLOAT_16_TYPE_H_
