//
// Created by sam on 7/8/24.
//

#include "scalar_vector.h"

using namespace rpy;
using namespace rpy::scalars;

VectorOperation::~VectorOperation() = default;

void VectorOperation::resize_destination(ScalarVector& arg, dimn_t new_size)
        const
{
    if (arg.dimension() < new_size) {
        arg.resize_dim(new_size);
    }
}
