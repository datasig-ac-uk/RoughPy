#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP

#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/common/scalars.hpp"

namespace rpy::compute::basic {
inline namespace v1 {


template <typename Context, typename ArgIter, typename Basis, typename Scalar>
void vector_inplace_scalar_multiply(
    Context const& ctx,
    DenseVectorView<ArgIter, Basis> arg, Scalar const& scalar)
{
    using ArgView = DenseVectorView<ArgIter, Basis>;
    using Index = typename ArgView::Index;

    for (Index i=0; i < arg.size(); ++i) {
        arg[i] *= scalar;
    }
}

template <typename ArgIter, typename Basis, typename Scalar>
void vector_inplace_scalar_multiply(DenseVectorView<ArgIter, Basis> arg, Scalar const& scalar)
{
    using Traits = scalars::Traits<typename DenseVectorView<ArgIter, Basis>::Scalar>;
    return vector_inplace_scalar_multiply(
        Traits{},
        std::move(arg),
        scalar
    );
}

} // version namespace
} // namespace rpy::compute::basic

#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP
