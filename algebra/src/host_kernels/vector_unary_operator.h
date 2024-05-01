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

template <template <typename> class Op, typename T>
class VectorUnaryOperator
{
    using operator_type = Op<T>;

public:
    void operator()(Slice<T> out, Slice<const T> in) const
    {
        operator_type op;
        for (auto&& [oscal, iscal] : views::zip(out, in)) { oscal = op(iscal); }
    }
};

template <template <typename...> class Op, typename T>
class VectorUnaryWithScalarOperator
{
    using operator_type = Op<T>;

public:
    void operator()(Slice<T> out, Slice<const T> in, const T& scal) const
    {
        operator_type op(scal);
        for (auto&& [oscal, iscal] : views::zip(out, in)) { oscal = op(iscal); }
    }
};

template <template <typename> class Op, typename T>
class VectorInplaceUnaryOperator
{
    using operator_type = Op<T>;

public:
    void operator()(Slice<T> arg) const
    {
        operator_type op;
        for (auto&& oscal : arg) { oscal = op(oscal); }
    }
};

template <template <typename...> class Op, typename T>
class VectorInplaceUnaryWithScalarOperator
{
    using operator_type = Op<T>;

public:
    void operator()(Slice<T> arg, const T& scal) const
    {
        operator_type op(scal);
        for (auto&& oscal : arg) { oscal = op(oscal); }
    }
};

}// namespace algebra
}// namespace rpy

#endif// VECTOR_UNARY_OPERATOR_H
