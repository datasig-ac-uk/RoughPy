#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_ADDITION_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_ADDITION_HPP


#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {


template <typename S, typename Basis, typename Op=ops::Identity>
void vector_inplace_addition(
    DenseVectorView<S*, Basis> lhs,
    DenseVectorView<S const*, Basis> rhs,
    Op&& op=Op{}
)
{
    using Size = typename DenseVectorView<S*, Basis>::Size;

    // We don't respect min_degree here, but we might want to add this support
    // later.

    auto const common_size = std::min(lhs.size(), rhs.size());

    for (Size i=0; i < common_size; ++i) {
        lhs[i] += op(rhs[i]);
    }

}

} // version namespace
} // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_ADDITION_HPP
