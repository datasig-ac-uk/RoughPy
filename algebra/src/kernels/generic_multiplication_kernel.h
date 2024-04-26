//
// Created by sam on 4/24/24.
//

#ifndef GENERIC_MUTIPLICATION_KERNEL_H
#define GENERIC_MUTIPLICATION_KERNEL_H

#include <roughpy/core/container/vector.h>

#include "arg_data.h"
#include "argument_specs.h"
#include "basis_key.h"
#include "generic_kernel.h"
#include "kernel.h"
#include "multiplication_impl.h"
#include "vector.h"

namespace rpy {
namespace algebra {
namespace dtl {

class GenericSquareMultiplicationKernel
{
    GenericBinaryFunction m_func;

    using generic_key_result_type
            = containers::SmallVec<pair<BasisKey, scalars::Scalar>, 1>;
    using key_func_type = std::function<
            generic_key_result_type(const BasisKey&, const BasisKey&)>;
    key_func_type m_key_func;
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
    explicit GenericSquareMultiplicationKernel(
            GenericBinaryFunction&& func,
            key_func_type&& key_func,
            const Basis* basis
    )
        : m_func(std::move(func)),
          m_key_func(std::move(key_func)),
          p_basis(basis)
    {}

    void
    operator()(VectorData& out, const VectorData& left, const VectorData& right)
            const;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_MUTIPLICATION_KERNEL_H
