#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_ADJOINT_MUL_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_ADJOINT_MUL_HPP

#include "roughpy_compute/common/basis.hpp"
#include "roughpy_compute/common/bitmask.hpp"
#include "roughpy_compute/dense/views.hpp"


namespace rpy::compute::basic {
inline namespace v1 {
template <typename OutIter, typename OpIter, typename ArgIter>
void st_adj_mul(
    DenseTensorView<OutIter> out,
    DenseTensorView<OpIter> op,
    DenseTensorView<ArgIter> arg
    )
{
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OpIter>::Index;

    using Mask = BitMask<Index>;

    CacheArray<int16_t, 32> letters(out.max_degree());
    const Index width = out.width();

    for (Degree func_degree = 0; func_degree <= func.max_degree(); ++func_degree) {
        const auto func_level = arg.at_level(func_degree);
        const auto size = func_level.size();

        for (Index i=0; i < size; ++i) {

            TensorBasis::unpack_index_to_letters(
                letters.data(),
                func_degree,
                i,
                width
                );

            Scalar scalar = func_level[i];

            for (Mask mask{}; mask <= Mask(func_degree); ++mask) {
                Index op_idx = 0;
                Degree  op_degree = 0;
                Index  out_idx = 0;
                Degree  out_degree = 0;

                TensorBasis::unpack_index_to_letters(
                    letters.data(),
                    func_degree,
                    width,
                    mask,
                    op_degree,
                    op_idx,
                    out_degree,
                    out_idx);

                auto op_level = op.at_level(op_degree);
                auto out_level = out.at_level(out_degree);

                out_level[out_idx] += scalar * op_level[op_idx];
            }

        }


    }

}


} // version namespace
} // namespace rpy::compute::basic


#endif// ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_ADJOINT_MUL_HPP
