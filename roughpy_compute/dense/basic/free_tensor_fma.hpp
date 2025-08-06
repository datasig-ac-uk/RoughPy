#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_FMA_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_FMA_HPP

#include <algorithm>

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {


template <typename OutIter, typename LhsIter, typename RhsIter, typename Op=ops::Identity>
void ft_fma(DenseTensorView<OutIter> out,
            DenseTensorView<LhsIter> lhs,
            DenseTensorView<RhsIter> rhs,
            Op&& op=Op{})
{
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;

    auto out_min_degree = std::max(Degree{1}, out.min_degree());

    for (Degree out_degree = out.max_degree(); out_degree >= out_min_degree; --
         out_degree) {
        auto const lhs_deg_max = std::min(lhs.max_degree(),
                                          out_degree - rhs.min_degree());
        auto const lhs_deg_min = std::max(lhs.min_degree(),
                                          out_degree - rhs.max_degree());

        auto out_frag = out.at_level(out_degree);

        for (Degree lhs_degree = lhs_deg_max; lhs_degree >= lhs_deg_min; --
             lhs_degree) {
            auto const rhs_degree = out_degree - lhs_degree;

            auto lhs_frag = lhs.at_level(lhs_degree);
            auto rhs_frag = rhs.at_level(rhs_degree);

            for (Index i = 0; i < lhs_frag.size(); ++i) {
                for (Index j = 0; j < rhs_frag.size(); ++j) {
                    out_frag[i * rhs_frag.size() + j] += op(
                        lhs_frag[i] * rhs_frag[j]);
                }
            }
        }
    }

    if (out.min_degree() == 0 && lhs.min_degree() == 0 && rhs.min_degree() == 0) {
        out[0] += op(lhs[0] * rhs[0]);
    }

}


}// version namespce
}// namespace rpy::compute::basic

#endif //ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_FMA_HPP