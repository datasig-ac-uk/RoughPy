#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/common/sparse_matrix.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {

template <
        typename OutIter,
        typename CompressedSparseMatrix,
        typename ArgIter,
        typename Op = ops::Identity>
[[gnu::always_inline]] inline void apply_sparse_linear_map(
        DenseVectorFragment<OutIter> out,
        CompressedSparseMatrix const& matrix,
        DenseVectorFragment<ArgIter> arg,
        Op&& op = {}
)
{
    using Index = typename CompressedSparseMatrix::difference_type;
    if constexpr (CompressedSparseMatrix::compressed_dim == CompressedRow) {
        const auto n_rows = matrix.rows();
        for (Index i = 0; i < n_rows; ++i) {
            auto const row = matrix.in_dim(i);
            auto& coeff = out[i];

            for (auto&& [j, c] : row){
                coeff += op(c * arg[j]);
            }
        }
    } else {
        const auto cols = matrix.cols();
        for (Index j = 0; j < cols; ++j) {
            const auto col = matrix.in_dim(j);

            for (auto&& [i, c] : col) {
                out[i] += op(c * arg[j]);
            }
        }
    }

    // auto const n_rows = matrix.n_rows();
    //
    // for (int32_t i=0; i < n_rows; ++i) {
    //     auto& coeff = out[i];
    //
    //     for (auto&& [j, c] : matrix.row(i)) {
    //         coeff += c * arg[j];
    //     }
    //
    // }
}

}// namespace v1
}// namespace rpy::compute::basic
#endif// ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP