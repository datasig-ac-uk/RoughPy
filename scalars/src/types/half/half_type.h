//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_


#include "scalar_helpers/standard_scalar_type.h"
#include "scalar_implementations/half.h"

#include "half_random_generator.h"

namespace rpy {
namespace scalars {

class HalfType : public dtl::StandardScalarType<Half>
{
    using base_t = dtl::StandardScalarType<Half>;


public:
    HalfType();

    static const ScalarType* get() noexcept;
};

RPY_LOCAL extern const HalfType half_type;


}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_
