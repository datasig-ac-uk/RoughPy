//
// Created by sam on 15/04/24.
//

#ifndef GENERIC_KERNEL_VS_H
#define GENERIC_KERNEL_VS_H

#include "arg_data.h"
#include "argument_specs.h"
#include "common.h"
#include "generic_kernel.h"

namespace rpy {
namespace algebra {
namespace dtl {

template <>
class GenericKernel<MutableVectorArg, ConstVectorArg, ConstVectorArg>
{
    GenericBinaryFunction m_func;
    const Basis* p_basis;

    void
    eval_sss(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_ssd(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_sds(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_sdd(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_dss(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_dsd(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_dds(VectorData& out, const VectorData& left, const VectorData& right)
            const;
    void
    eval_ddd(VectorData& out, const VectorData& left, const VectorData& right)
            const;

public:
    explicit GenericKernel(GenericBinaryFunction&& func, const Basis* basis)
        : m_func(std::move(func)),
          p_basis(basis)
    {}

    void operator()(VectorData& out, const VectorData& left, const VectorData& right) const;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_KERNEL_VS_H
