#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_ADDITION_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_ADDITION_HPP


#include <cassert>

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"


namespace rpy::compute::basic {
inline namespace v1 {

template <typename S, typename Basis, typename LhsOp=ops::Identity, typename RhsOp=ops::Identity>
void vector_addition(
    DenseVectorView<S*, Basis> out,
    DenseVectorView<S const*, Basis> lhs,
    DenseVectorView<S const*, Basis> rhs,
    LhsOp&& lhs_op=LhsOp{},
    RhsOp&& rhs_op=RhsOp{}
)
{
    using Degree = typename DenseVectorView<S*, Basis>::Degree;
    using Size = typename DenseVectorView<S*, Basis>::Size;


    // I'm going to ignore minimum degree at the moment. We might want to
    // fix this later

    auto common_size = std::min(lhs.size(), rhs.size());

    for (Size i=0; i < std::min(out.size(), common_size); ++i) {
        out[i] = lhs_op(lhs[i]) + rhs_op(rhs[i]);
    }

    for (Size i=common_size; i < std::min(out.size(), lhs.size()); ++i) {
        out[i] = lhs_op(lhs[i]);
    }

    for (Size i=common_size; i < std::min(out.size(), rhs.size()); ++i) {
        out[i] = rhs_op(rhs[i]);
    }

}

} // version namespace
} // namespace rpy::compute::basic

#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_ADDITION_HPP
