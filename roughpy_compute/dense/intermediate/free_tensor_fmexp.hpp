#ifndef ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
#define ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP

#include <algorithm>

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/common/scalars.hpp"

#include "roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp"
#include "roughpy_compute/dense/basic/vector_inplace_addition.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::intermediate {
inline namespace v1 {

template <
        typename Context,
        typename OutIter,
        typename MulIter,
        typename ExpIter>
void ft_fmexp(
        Context const& ctx,
        DenseTensorView<OutIter> out,
        DenseTensorView<MulIter> multiplier,
        DenseTensorView<ExpIter> exponent
)
{
    using OutView = DenseTensorView<OutIter>;
    using Degree = typename OutView::Degree;
    using Scalar = typename OutView::Scalar;

    auto const common_size = std::min(out.size(), multiplier.size());
    std::copy_n(multiplier.data(), common_size, out.data());

    const auto one = ctx.one();

    auto const max_degree = out.max_degree();
    for (Degree deg = max_degree; deg > 0; --deg) {
        auto const max_level = max_degree - deg + 1;

        basic::ft_inplace_mul(
                ctx,
                out.truncate(max_level),
                exponent.truncate(max_level, 1),
                ops::RightMultiplyBy<Scalar>(one / deg)
        );

        basic::vector_inplace_addition(ctx, out, multiplier);
    }
}

template <typename OutIter, typename MulIter, typename ExpIter>
void ft_fmexp(
        DenseTensorView<OutIter> out,
        DenseTensorView<MulIter> multiplier,
        DenseTensorView<ExpIter> exponent
)
{
    using Traits = scalars::Traits<typename DenseTensorView<OutIter>::Scalar>;
    return ft_fmexp(
            Traits{},
            std::move(out),
            std::move(multiplier),
            std::move(exponent)
    );
}

}// namespace v1
}// namespace rpy::compute::intermediate

#endif// ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
