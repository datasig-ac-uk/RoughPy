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


    for (Degree op_degree=0; op_degree <= op.max_degree(); ++op_degree) {

    }


}
}// version namespace
}// namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ADJOINT_LEFT_MUL_HPP