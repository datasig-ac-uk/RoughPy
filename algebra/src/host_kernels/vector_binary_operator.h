//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_BINARY_OPERATOR_H
#define VECTOR_BINARY_OPERATOR_H

#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {

template <typename T, typename Op>
class VectorBinaryOperator
{
public:
    void
    operator()(Slice<T> out, Slice<const T> left, Slice<const T> right) const
    {
        Op op;
        for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
            out = op(left, right);
        }
    }
};

template <typename T, typename Op>
class VectorBinaryWithScalarOperator
{
public:
    void operator()(
            Slice<T> out,
            Slice<const T> left,
            Slice<const T> right,
            const T& scal
    ) const
    {
        Op op(scal);
        for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
            out = op(left, right);
        }
    }
};

template <typename T, typename Op>
class VectorInplaceBinaryOperator
{
public:
    void operator()(Slice<T> left, Slice<const T> right) const
    {
        Op op;
        for (auto& [oscal, rscal] : views::zip(left, right)) {
            oscal = op(oscal, rscal);
        }
    }
};

template <typename T, typename Op>
class VectorInplaceBinaryWithScalarOperator
{
public:
    void operator()(Slice<T> left, Slice<const T> right, const T& scal) const
    {
        Op op(scal);
        for (auto& [oscal, rscal] : views::zip(left, right)) {
            oscal = op(oscal, rscal);
        }
    }
};

}// namespace algebra
}// namespace rpy

#endif// VECTOR_BINARY_OPERATOR_H
