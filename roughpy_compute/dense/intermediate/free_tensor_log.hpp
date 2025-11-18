#ifndef ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_LOG_HPP
#define ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_LOG_HPP

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/common/scalars.hpp"
#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp"

namespace rpy::compute::intermediate {
inline namespace v1 {

template <typename Context, typename OutIter, typename ArgIter>
void ft_log(
    Context const& ctx,
    DenseTensorView<OutIter> out,
    DenseTensorView<ArgIter> arg)
{
    using OutView = DenseTensorView<OutIter>;
    using Degree = typename OutView::Degree;

    const auto unit = ctx.one();

    auto const max_degree = out.max_degree();
    for (Degree deg=max_degree; deg > 0; --deg) {
        auto const max_level = max_degree - deg + 1;

        if (deg % 2 == 0) {
            out[0] -= unit / deg;
        } else {
            out[0] += unit / deg;
        }

        basic::ft_inplace_mul(
            ctx,
            out.truncate(max_level),
            arg.truncate(max_level, 1)
        );

    }
}


template <typename OutIter, typename ArgIter>
void ft_log(
    DenseTensorView<OutIter> out,
    DenseTensorView<ArgIter> arg)
{
    using Traits = scalars::Traits<typename DenseTensorView<OutIter>::Scalar>;
    return ft_log(Traits{}, std::move(out), std::move(arg));
}

} // version namespace
} // namespace rpy::compute::intermediate



#endif //ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_LOG_HPP
