//
// Created by sam on 15/04/24.
//

#ifndef GENERIC_KERNEL_VS_H
#define GENERIC_KERNEL_VS_H

#include "common.h"
#include "generic_kernel.h"
#include "argument_specs.h"
#include "arg_data.h"

#include <functional>


namespace rpy { namespace algebra { namespace dtl {

template <>
class GenericKernel<MutableVectorArg, ConstScalarArg>
{

    GenericUnaryFunction m_func;

public:

    explicit GenericKernel(GenericUnaryFunction&& arg) : m_func(std::move(arg)) {}

    void operator()(VectorData& data, const scalars::Scalar& scalar) const;

};

}}}

#endif //GENERIC_KERNEL_VS_H
