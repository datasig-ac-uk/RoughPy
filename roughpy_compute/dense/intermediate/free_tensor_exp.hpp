#ifndef ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_EXP_HPP
#define ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_EXP_HPP



#include "roughpy_compute/commmon/operations.hpp"

#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/dense/basic/free_tensor_inplace_mul.hpp"

namespace rpy::compute::intermediate {
inline namespace v1 {

template <typename S>
void ft_exp(DenseTensorView<S*> out, DenseTensorView<S const*> arg)
{
    using Degree = typename DenseTensorView<S*>::Degree;

    constexpr S unit { 1 };


    out[0] = unit;

    auto const max_degree = out.max_degree();
    for (Degree deg=max_degree(); deg > 0; --deg) {
        auto const max_level = max_degree - deg + 1;

        basic::ft_inplace_mul(
            out.truncate(max_level),
            arg.truncate(max_level, 1);
            ops::DivideBy<S>(static_cast<S>(deg))
        );

        out[0] += unit;
    }
}

} // version namespace
} // namespace rpy::compute::intermediate

#endif //ROUGHPY_COMPUTE_DENSE_INTERMEDIATE_FREE_TENSOR_EXP_HPP
