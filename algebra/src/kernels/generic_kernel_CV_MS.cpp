//
// Created by sam on 15/04/24.
//

#include "generic_kernel_CV_MS.h"

void rpy::algebra::dtl::GenericKernel<
        rpy::algebra::dtl::ConstVectorArg,
        rpy::algebra::dtl::MutableScalarArg>::
operator()(const VectorData& data, scalars::Scalar& scalar) const
{
    for (size_t i = 0; i < data.size(); ++i) {
        m_func(scalar, data.scalars()[i]);
    }
}
