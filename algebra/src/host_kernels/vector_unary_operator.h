//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_UNARY_OPERATOR_H
#define VECTOR_UNARY_OPERATOR_H

#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {

template <typename Op>
class VectorUnaryOperator
{
public:
    void operator()(Slice<T> out, Slice<const T> in)
    {
        Op op;
        for (auto&& [oscal, iscal] : views::zip(out, in)) { oscal = op(iscal); }
    }
};

template <typename T, typename Op>
class VectorUnaryWithScalarOperator
{
public:
    void operator()(Slice<T> out, Slice<const T> in, const T& scal)
    {
        Op op(scal);
        for (auto&& [oscal, iscal] : views::zip(out, in)) { oscal = op(iscal); }
    }
};

template <typename T, typename Op>
class VectorUnaryInplaceOperator
{
public:
    void operator()(Slice<T> arg)
    {
        Op op;
        for (auto& oscal : arg) { oscal = op(oscal); }
    }
};

template <typename T, typename Op>
class VectorInplaceUnaryWithScalarOperator
{
public:
    void operator()(Slice<T> arg, const T& scal)
    {
        Op op(scal);
        for (auto& oscal : arg) { oscal = op(oscal); }
    }
};

}// namespace algebra
}// namespace rpy

#endif// VECTOR_UNARY_OPERATOR_H
