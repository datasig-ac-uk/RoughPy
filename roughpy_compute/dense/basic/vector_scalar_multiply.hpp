#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP

#include "roughpy_compute/common/scalars.hpp"

namespace rpy::compute::basic {
inline namespace v1 {

template <typename Context, typename S, typename Basis>
void vector_scalar_multiply(
    Context const& ctx,
    DenseVectorView<S*, Basis> out,
    DenseVectorView<S const*, Basis> in,
    S const& scalar)
{
    using Index = typename DenseVectorView<S*, Basis>::Index;

    auto const common_size = std::min(out.size(), in.size());

    for (Index i=0; i < common_size; ++i) {
        out[i] = in[i] * scalar;
    }

}

template <typename S, typename Basis>
void vector_scalar_multiply(
    DenseVectorView<S*, Basis> out,
    DenseVectorView<S const*, Basis> in,
    S const& scalar)
{
    using Traits = scalars::Traits<typename DenseVectorView<S*, Basis>::Scalar>;

    return vector_scalar_multiply(
        Traits{},
        std::move(out), std::move(in), scalar);
}

} // version namespace
}; // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_SCALAR_MULTIPLY_HPP
