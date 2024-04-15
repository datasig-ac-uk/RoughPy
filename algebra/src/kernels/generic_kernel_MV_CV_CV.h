//
// Created by sam on 15/04/24.
//

#ifndef GENERIC_KERNEL_VS_H
#define GENERIC_KERNEL_VS_H

#include "common.h"
#include "generic_kernel.h"
#include "argument_specs.h"
#include "arg_data.h"


namespace rpy { namespace algebra { namespace dtl {

template <>
class GenericKernel<MutableVectorArg, ConstVectorArg, ConstVectorArg>
{
    GenericBinaryFunction m_func;

public:

    explicit GenericKernel(GenericBinaryFunction&& func) : m_func(std::move(func))
    {}

    void operator()(VectorData& out, VectorData& left, VectorData& right) const;

};

}}}


#endif //GENERIC_KERNEL_VS_H
