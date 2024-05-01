//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_BINARY_OPERATOR_H
#define VECTOR_BINARY_OPERATOR_H

#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace algebra {

template <template <typename> class Op, typename T>
class VectorBinaryOperator
{
    using operator_type = Op<T>;
public:
    void
    operator()(Slice<T> out, Slice<const T> left, Slice<const T> right) const
    {
        operator_type op;
        for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
            oscal = op(lscal, rscal);
        }
    }
};

template <template <typename...> class Op, typename T>
class VectorBinaryWithScalarOperator
{
    using operator_type = Op<T>;
public:
    void operator()(
            Slice<T> out,
            Slice<const T> left,
            Slice<const T> right,
            const T& scal
    ) const
    {
        operator_type op(scal);
        for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
            oscal = op(lscal, rscal);
        }
    }
};

template <template <typename> class Op, typename T>
class VectorInplaceBinaryOperator
{
    using operator_type = Op<T>;
public:
    void operator()(Slice<T> left, Slice<const T> right) const
    {
        operator_type op;
        for (auto&& [oscal, rscal] : views::zip(left, right)) {
            oscal = op(oscal, rscal);
        }
    }
};

template <template <typename...> class Op, typename T>
class VectorInplaceBinaryWithScalarOperator
{
    using operator_type = Op<T>;
public:
    void operator()(Slice<T> left, Slice<const T> right, const T& scal) const
    {
        operator_type op(scal);
        for (auto&& [oscal, rscal] : views::zip(left, right)) {
            oscal = op(oscal, rscal);
        }
    }
};

}// namespace algebra
}// namespace rpy

#endif// VECTOR_BINARY_OPERATOR_H
