#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_INPLACE_MUL_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_INPLACE_MUL_HPP

#include <algorithm>

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"


namespace rpy::compute::basic {
inline namespace v1 {

template <typename LhsIter, typename RhsIter, typename Op=ops::Identity>
void ft_inplace_mul(DenseTensorView<LhsIter> lhs, DenseTensorView<RhsIter> rhs, Op&& op=Op{}) {

    using Degree = typename DenseTensorView<LhsIter>::Degree;
    using Index = typename DenseTensorView<RhsIter>::Index;
    using Scalar = typename DenseTensorView<LhsIter>::Scalar;


    for (Degree out_degree = lhs.max_degree();
         out_degree >= std::max(Degree{1}, lhs.min_degree());
         --out_degree) {

        const auto lhs_deg_min = std::max(lhs.min_degree(), out_degree - rhs.max_degree());

        // Normally we'd want the min of lhs_depth and out_deg, but lhs_depth >= out_deg
        // always, so we can remove the min. We don't want the main loop to consume the
        // out_deg by 0 computation, so we subtract one from the max degree and handle
        // that first. We also want to avoid any terms that would involve data from
        // rhs of degree less than rhs_min_degree.
        const auto lhs_deg_max = out_degree - std::max(Degree{1}, rhs.min_degree());


        auto out_frag = lhs.at_level(out_degree);

        /*
         * The only computation that involves out_frag on the right-hand side of
         * the computation is the one where we compute out_frag * rhs[0], where
         * rhs[0] can be implictly zero if min_degree > 0. From this point on,
         * if we use out_frag on the right-hand side anywhere then we will get
         * incorrect results. The results of products of lower-degree terms will
         * be accumulated into this out_frag after this first assign operation.
         */
        if (rhs.min_degree() == 0) {
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = op(out_frag[i] * rhs[0]);
            }
        } else {
            // if rhs_min_degree > 0 then the unit is implicitly zero, so the
            // lhs unit should be set to zero too.
            for (Index i = 0; i < out_frag.size(); ++i) {
                out_frag[i] = Scalar { 0 };
            }
        }


        for (Degree lhs_degree = lhs_deg_max; lhs_degree >= lhs_deg_min; --lhs_degree) {
            const auto rhs_degree = out_degree - lhs_degree;

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


    /*
     * We carefully avoided this case in the loop above. The final step is to
     * compute the new unit element of lhs. This is either the product of the
     * two units as they exist or implicitly zero. We only treat the cases where
     * lhs.min_degree() == 0. Otherwise we might actually write to a location
     * that we don't have any right to access. I don't think this case ever
     * actually appears naturally.
     */
    if (lhs.min_degree() == 0 && rhs.min_degree() == 0) {
        lhs[0] = op(lhs[0] * rhs[0]);
    } else if (lhs.min_degree() == 0) {
        lhs[0] = Scalar { 0 };
    }
}


} // version namespace
} // namespace rpy::compute::basic


#endif //ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_INPLACE_MUL_HPP
