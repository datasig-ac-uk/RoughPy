//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_


#include "scalar_helpers/standard_scalar_type.h"
#include "scalar_types.h"

#include "half_random_generator.h"

namespace rpy {
namespace scalars {

class HalfType : public dtl::StandardScalarType<half>
{
    using base_t = dtl::StandardScalarType<half>;


public:
    HalfType();

    static const ScalarType* get() noexcept;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_HALF_HALF_TYPE_H_
