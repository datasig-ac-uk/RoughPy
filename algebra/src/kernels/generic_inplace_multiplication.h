//
// Created by sam on 4/26/24.
//

#ifndef GENERIC_INPLACE_MULTIPLICATION_H
#define GENERIC_INPLACE_MULTIPLICATION_H

#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "basis_key.h"
#include "generic_kernel.h"
#include "vector.h"

namespace rpy {
namespace algebra {
namespace dtl {

class GenericInplaceMultiplication
{
    using generic_key_result_type
            = containers::SmallVec<pair<BasisKey, scalars::Scalar>, 1>;
    using key_func_type = std::function<
            generic_key_result_type(const BasisKey&, const BasisKey&)>;

    GenericUnaryFunction m_scalar_func;
    key_func_type m_key_func;
    const Basis* p_basis;

public:
    explicit GenericInplaceMultiplication(
            GenericUnaryFunction&& func,
            key_func_type&& key_func,
            const Basis* basis
    )
        : m_scalar_func(std::move(func)),
          m_key_func(std::move(key_func)),
          p_basis(basis)
    {}

private:
    void eval_ss(VectorData& out, const VectorData& arg) const;
    void eval_sd(VectorData& out, const VectorData& arg) const;
    void eval_ds(VectorData& out, const VectorData& arg) const;
    void eval_dd(VectorData& out, const VectorData& arg) const;

public:
    void operator()(VectorData& out, const VectorData& arg) const;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_INPLACE_MULTIPLICATION_H
