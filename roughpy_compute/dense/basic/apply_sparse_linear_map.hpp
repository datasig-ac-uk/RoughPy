#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP

#include "roughpy_compute/common/sparse_matrix.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {

template <typename Scalar, typename OutIter, typename ArgIter>
[[gnu::always_inline]] inline
void apply_sparse_linear_map(DenseVectorFragment<OutIter> out, CompressedRowMatrix<Scalar> const& matrix, DenseVectorFragment<ArgIter> arg)
{

    auto const n_rows = matrix.n_rows();

    for (int32_t i=0; i < n_rows; ++i) {
        auto& coeff = out[i];

        for (auto&& [j, c] : matrix.row(i)) {
            coeff += c * arg[j];
        }

    }
}

} // version namespace
} // namespace rpy::compute::basic
#endif //ROUGHPY_COMPUTE_DENSE_BASIC_APPLY_SPARSE_LINEAR_MAP_HPP