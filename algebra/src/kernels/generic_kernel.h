//
// Created by sam on 15/04/24.
//

#ifndef GENERIC_KERNEL_H
#define GENERIC_KERNEL_H

#include "arg_data.h"
#include "common.h"

#include <roughpy/scalars/scalar.h>

#include <functional>

namespace rpy {
namespace algebra {
namespace dtl {

using GenericUnaryFunction
        = std::function<void(scalars::Scalar&, const scalars::Scalar&)>;
using GenericBinaryFunction = std::function<
        void(scalars::Scalar&, const scalars::Scalar&, const scalars::Scalar&)>;
using GenericTernaryFunction = std::function<
        void(scalars::Scalar&,
             const scalars::Scalar&,
             const scalars::Scalar&,
             const scalars::Scalar&)>;

inline int get_sparse_dense_config(const VectorData& arg) noexcept
{
    return (arg.sparse() ? 0 : 1);
}

template <typename... Rem>
RPY_NO_DISCARD int get_sparse_dense_config(
        const VectorData& first,
        const Rem&... remaining
) noexcept
{
    return (first.sparse() ? 0 : 1 << sizeof...(remaining))
            | get_sparse_dense_config(remaining...);
}

class DenseVectorMap
{
    scalars::ScalarArray& m_array;
    const Basis* p_basis;

public:
    DenseVectorMap(scalars::ScalarArray& array, const Basis* basis)
        : m_array(array),
          p_basis(basis)
    {}

    scalars::Scalar operator[](const BasisKey& key)
    {
        return m_array[p_basis->to_index(key)];
    }
};

template <>
class GenericKernel<MutableVectorArg, ConstScalarArg>;

template <>
class GenericKernel<ConstVectorArg, MutableScalarArg>;

template <>
class GenericKernel<MutableVectorArg, ConstVectorArg>;

template <>
class GenericKernel<MutableVectorArg, ConstVectorArg, ConstScalarArg>;

template <>
class GenericKernel<MutableVectorArg, ConstVectorArg, ConstVectorArg>;

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// GENERIC_KERNEL_H
