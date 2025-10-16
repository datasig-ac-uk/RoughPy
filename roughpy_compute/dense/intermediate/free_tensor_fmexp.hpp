#ifndef ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
#define ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP

#include <algorithm>

#include "roughpy_compute/common/operations.hpp"

#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp"
#include "roughpy_compute/dense/basic/vector_inplace_addition.hpp"

namespace rpy::compute::intermediate {
inline namespace v1 {

template <typename OutIter, typename MulIter, typename ExpIter>
void ft_fmexp(
        DenseTensorView<OutIter> out,
        DenseTensorView<MulIter> multiplier,
        DenseTensorView<ExpIter> exponent)
{
    using OutView = DenseTensorView<OutIter>;
    using Degree = typename OutView::Degree;
    using Scalar = typename OutView::Scalar;

    auto const common_size = std::min(out.size(), multiplier.size());
    std::copy_n(multiplier.data(), common_size, out.data());

    auto const max_degree = out.max_degree();
    for (Degree deg=max_degree; deg > 0; --deg) {
        auto const max_level = max_degree - deg + 1;

        basic::ft_inplace_mul(
            out.truncate(max_level),
            exponent.truncate(max_level, 1),
            ops::DivideBy<Scalar>(static_cast<Scalar>(deg))
        );

        basic::vector_inplace_addition(out, multiplier);
    }
}

} // version namespace
} // namespace rpy::compute::intermediate

#endif //ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
