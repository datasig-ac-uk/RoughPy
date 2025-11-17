#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_PRODUCT_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_PRODUCT_HPP

#include <algorithm>

#include "roughpy_compute/common/basis.hpp"
#include "roughpy_compute/common/bitmask.hpp"
#include "roughpy_compute/common/cache_array.hpp"
#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/common/scalars.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {
/*
 * The version 1 implementation is a straightforward, untiled algorithm. This
 * takes each output index, breaks into letters and reshuffles them according to
 * a bitmask to obtain all the contributors to the coefficient of that output
 * scalar. This is written to the output.
 */

template <
        typename Context,
        typename OutIter,
        typename LhsIter,
        typename RhsIter,
        typename Op = ops::Identity>
void st_fma(
        Context const& ctx,
        DenseTensorView<OutIter> out,
        DenseTensorView<LhsIter> lhs,
        DenseTensorView<RhsIter> rhs,
        Op&& op = {}
)
{
    using Scalar = typename DenseTensorView<OutIter>::Scalar;
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;
    using Mask = BitMask<Index>;

    const Index width = out.width();

    CacheArray<int16_t, 32> letters(out.max_degree());

    out[0] += op(lhs[0] * rhs[0]);
    Index out_size = width;

    for (Degree out_deg = 1; out_deg <= lhs.max_degree(); ++out_deg) {
        auto out_level = out.at_level(out_deg);

        for (Index i = 0; i < out_size; ++i) {
            // unpack the outer letters
            TensorBasis::unpack_index_to_letters(
                    letters.data(),
                    out_deg,
                    i,
                    width
            );

            Scalar acc{0};

            for (Mask mask{}; mask <= Mask(out_deg); ++mask) {
                Index lhs_idx = 0;
                Degree lhs_degree = 0;
                Index rhs_idx = 0;
                Degree rhs_degree = 0;

                TensorBasis::pack_masked_index(
                        letters.data(),
                        out_deg - 1,
                        width,
                        mask,
                        lhs_degree,
                        lhs_idx,
                        rhs_degree,
                        rhs_idx
                );

                auto lhs_level = lhs.at_level(lhs_degree);
                auto rhs_level = rhs.at_level(rhs_degree);
                acc += lhs_level[lhs_idx] * rhs_level[rhs_idx];
            }

            out_level[i] += op(acc);
        }

        out_size *= width;
    }
}

template <
        typename OutIter,
        typename LhsIter,
        typename RhsIter,
        typename Op = ops::Identity>
void st_fma(
        DenseTensorView<OutIter> out,
        DenseTensorView<LhsIter> lhs,
        DenseTensorView<RhsIter> rhs,
        Op&& op = {}
)
{
  using Traits = scalars::Traits<typename DenseTensorView<OutIter>::Scalar>;
  return st_fma(
          Traits{},
          std::move(out),
          std::move(lhs),
          std::move(rhs),
          std::forward<Op>(op)
  );
}

}// namespace v1

namespace v2 {
/*
 * The version 2 implementation uses a single letter tiling on the right
 * (fastest changing index) to group computations. This means we reduce the
 * number of index decompositions and reshufflings that must be performed by a
 * factor of w. More importantly, we improve the cache locality of all accesses
 * by gathering blocks of data instead of single elements.
 *
 * This implementation is originally due to Mike Giles.
 */

template <
        typename Context,
        typename OutIter,
        typename LhsIter,
        typename RhsIter,
        typename Op = ops::Identity>
void st_fma(
        Context const& ctx,
        DenseTensorView<OutIter> out,
        DenseTensorView<LhsIter> lhs,
        DenseTensorView<RhsIter> rhs,
        Op&& op = {}
)
{
    using Scalar = typename DenseTensorView<OutIter>::Scalar;
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;
    using Mask = BitMask<Index>;

    const Index width = out.width();
    const Index tile_size = width;

    CacheArray<int16_t, 32> letters(out.max_degree());
    CacheArray<Scalar, 8> tile(tile_size);

    out[0] += op(lhs[0] * rhs[0]);
    Index out_size = 1;

    for (Degree out_deg = 1; out_deg <= lhs.max_degree(); ++out_deg) {
        auto out_level = out.at_level(out_deg);

        for (Index i = 0; i < out_size; ++i) {

            // unpack the outer letters
            TensorBasis::unpack_index_to_letters(
                    letters.data(),
                    out_deg - 1,
                    i,
                    width
            );

            std::fill_n(tile.data(), width, Scalar{0});

            for (Mask mask{}; mask < Mask(out_deg - 1); ++mask) {

                Index lhs_idx = 0;
                Degree lhs_degree = 0;
                Index rhs_idx = 0;
                Degree rhs_degree = 0;

                TensorBasis::pack_masked_index(
                        letters.data(),
                        out_deg - 2,
                        lhs_degree,
                        lhs_idx,
                        rhs_degree,
                        rhs_idx
                );

                auto lhs_level = lhs.at_level(lhs_degree);
                auto lhs_p1_level = lhs.at_level(lhs_degree + 1);
                auto rhs_level = rhs.at_level(rhs_degree);
                auto rhs_p1_level = rhs.at_level(rhs_degree + 1);

                for (Index j = 0; j < width; ++j) {
                    tile[j] += lhs_level[lhs_idx]
                            * rhs_p1_level[rhs_idx * width + j];
                    tile[j] += lhs_p1_level[lhs_idx * width + j]
                            * rhs_level[rhs_idx];
                }
            }

            // write out the results
            for (Index j = 0; j < width; ++j) {
                out_level[i * width + j] += op(tile[j]);
            }
        }

        out_size *= width;
    }
}

template <
        typename OutIter,
        typename LhsIter,
        typename RhsIter,
        typename Op = ops::Identity>
void st_fma(
        DenseTensorView<OutIter> out,
        DenseTensorView<LhsIter> lhs,
        DenseTensorView<RhsIter> rhs,
        Op&& op = {}
)
{
    using Traits = scalars::Traits<typename DenseTensorView<OutIter>::Scalar>;
    return st_fma(
            Traits{},
            std::move(out),
            std::move(lhs),
            std::move(rhs),
            std::forward<Op>(op)
    );
}

}// namespace v2
}// namespace rpy::compute::basic

#endif// ROUGHPY_COMPUTE_DENSE_BASIC_SHUFFLE_TENSOR_PRODUCT_HPP