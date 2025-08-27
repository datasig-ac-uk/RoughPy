#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP


#include "roughpy_compute/dense/views.hpp"


namespace rpy::compute::basic {
inline namespace v1 {

template <typename OutIter, typename OpIter, typename ArgIter>
void ft_adj_lmul(DenseTensorView<OutIter> out,
                 DenseTensorView<OpIter> op,
                 DenseTensorView<ArgIter> arg)
{
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;

    auto out_max_degree = out.max_degree();
    auto out_min_degree = out.min_degree();

    for (Degree out_degree=out_max_degree; out_degree >= out_min_degree; --out_degree) {
        auto op_min_degree = std::max(Degree{0}, out_degree - op.max_degree());
        auto op_max_degree = std::min(out_degree, op.max_degree() - arg.min_degree());


        for ()


    }


}
}// version namespace
}// namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP