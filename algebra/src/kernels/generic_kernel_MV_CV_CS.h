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
class GenericKernel<MutableVectorArg, ConstVectorArg, ConstScalarArg>
{
    GenericBinaryFunction m_func;
    const Basis* p_basis;

    void eval_sparse_sparse(
            VectorData& out,
            const VectorData& arg,
            scalars::ScalarCRef scal
    ) const;
    void eval_sparse_dense(
            VectorData& out,
            const VectorData& arg,
            scalars::ScalarCRef scal
    ) const;
    void eval_dense_sparse(
            VectorData& out,
            const VectorData& arg,
            scalars::ScalarCRef scal
    ) const;
    void eval_dense_dense(
            VectorData& out,
            const VectorData& arg,
            scalars::ScalarCRef scal
    ) const;

public:
    explicit
    GenericKernel(GenericBinaryFunction&& func, const Basis* basis = nullptr)
        : m_func(std::move(func)),
          p_basis(basis)
    {}

    void operator()(
            VectorData& out,
            const VectorData& arg,
            scalars::ScalarCRef scal
    ) const;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_KERNEL_VS_H
