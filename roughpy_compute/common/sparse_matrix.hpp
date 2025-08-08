#ifndef ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP
#define ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP

#include <cstdint>
#include <iterator>
#include <utility>

namespace rpy::compute {

template <typename Scalar>
class CompressedRowMatrix;


template <typename Scalar>
class IndexScalarIterator
{
    using Index = std::ptrdiff_t;

    Index const* indices_ = nullptr;
    Scalar const* values_ = nullptr;

    template <typename S>
    friend class CompressedRowMatrix;

public:
    using value_type = std::pair<Index, Scalar>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::forward_iterator_tag;

    constexpr IndexScalarIterator(Index const* indices, Scalar const* values)
        : indices_(indices), values_(values) {}

    constexpr auto operator*() const noexcept
    {
        return std::make_pair(*indices_, *values_);
    }

    constexpr IndexScalarIterator& operator++() noexcept
    {
        ++indices_;
        ++values_;
        return *this;
    }

    [[nodiscard]]
    constexpr IndexScalarIterator operator++(int) noexcept
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }


    friend constexpr bool operator==(
        IndexScalarIterator const& lhs,
        IndexScalarIterator const& rhs) noexcept
    {
        // We only need to check the indices pointers, since these
        // are incremented together
        return lhs.indices_ == rhs.indices_;
    }

    friend constexpr bool operator!=(
        IndexScalarIterator const& lhs,
        IndexScalarIterator const& rhs) noexcept { return !(lhs == rhs); }
};

template <typename Scalar>
class IndexScalarRange
{
public:
    using iterator = IndexScalarIterator<Scalar>;
    using const_iterator = iterator;
    using Index = std::ptrdiff_t;

private:
    iterator begin_;
    iterator end_;

public:
    constexpr IndexScalarRange(iterator begin, iterator end)
        : begin_(begin), end_(end) {}

    constexpr iterator begin() const noexcept { return begin_; }
    constexpr iterator end() const noexcept { return end_; }
};

template <typename Scalar>
class CompressedRowMatrix
{
    using Index = std::ptrdiff_t;
    int32_t n_rows_;
    int32_t n_cols_;

    Index const* row_offsets_;// size = n_rows + 1
    Index const* col_indices_;// size = n_nonzeros
    Scalar* values_;// size = n_nonzeros


public:
    constexpr CompressedRowMatrix(int32_t n_rows,
                                  int32_t n_cols,
                                  Index const* row_offsets,
                                  Index const* col_indices,
                                  Scalar* values)
        : n_rows_(n_rows), n_cols_(n_cols),
          row_offsets_(row_offsets), col_indices_(col_indices),
          values_(values) {}


    constexpr int32_t n_rows() const noexcept { return n_rows_; }
    constexpr int32_t n_cols() const noexcept { return n_cols_; }

    constexpr Index n_nonzeros() const noexcept
    {
        return row_offsets_[n_rows_];
    }

    constexpr IndexScalarRange<Scalar> row_data(int32_t row) const noexcept
    {
        return {
                IndexScalarIterator<Scalar>(
                    col_indices_ + row_offsets_[row],
                    values_ + row_offsets_[row]),
                IndexScalarIterator<Scalar>(
                    col_indices_ + row_offsets_[row + 1],
                    values_ + row_offsets_[row + 1])};
    }

};


}// namespace rpy::compute


#endif //ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP