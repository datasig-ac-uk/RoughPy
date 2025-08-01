#ifndef ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
#define ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP

#include <algorithm>

#include "roughpy_compute/commmon/operations.hpp"

#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp"
#include "roughpy_compute/dense/basic/free_tensor_inplace_add.hpp"

namespace rpy::compute::intermediate {
inline namespace v1 {

template <typename S>
void ft_fmexp(
        DenseTensorView<S*> out,
        DenseTensorView<S const*> multiplier,
        DenseTensorView<S const*> exponent)
{
    using Degree = typename DenseTensorView<S*>::Degree;

    auto const common_size = std::min(out.size(), multiplier.size());
    std::copy_n(multiplier.data(), common_size, out.data());

    auto const max_degree = out.max_degree();
    for (Degree deg=max_degree; deg > 0; --deg) {
        auto const max_level = max_degree - deg + 1;

        basic::ft_inplace_mul(
            out.truncate(max_level),
            exponent.truncate(max_level, 1),
            ops::DivideBy<S>(static_cast<S>(deg))
        );

        basic::inplace_add(out, multiplier);
    }
}

} // version namespace
} // namespace rpy::compute::intermediate

#endif //ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_FMEXP_HPP
