//
// Created by sam on 15/04/24.
//

#include "generic_kernel_MV_CS.h"

void rpy::algebra::dtl::GenericKernel<
        rpy::algebra::dtl::MutableVectorArg,
        rpy::algebra::dtl::ConstScalarArg>::
operator()(VectorData& data, scalars::ScalarCRef scalar) const
{
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto tmp = data.mut_scalars()[i];
        m_func(tmp, scalar);
    }
}
