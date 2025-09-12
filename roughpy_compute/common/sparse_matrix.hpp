#ifndef ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP
#define ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

#include "scalars.hpp"

namespace rpy::compute {

namespace dtl {


template <typename DataIter_, typename IndicesIter_>
class CompressedRangeIterator
{
    using DataTraits = std::iterator_traits<DataIter_>;
    using IndicesTraits = std::iterator_traits<IndicesIter_>;
public:
    using difference_type = std::common_type_t<
        typename DataTraits::difference_type,
        typename IndicesTraits::difference_type>;

    using Scalar = typename DataTraits::value_type;
    using Index = typename IndicesTraits::value_type;

    using value_type = std::pair<IndicesIter_ , DataIter_>;

    using reference = std::pair<
        typename IndicesTraits::reference,
        typename DataTraits::reference
    >;

    using iterator_category = std::forward_iterator_tag;

    struct Sentinel
    {
        difference_type end_index_;
    };

private:
    DataIter_ data_;
    IndicesIter_ indices_;
    difference_type index_;

public:

    constexpr CompressedRangeIterator(DataIter_ data, IndicesIter_ indices, difference_type index=0)
        : data_(std::move(data)), indices_(std::move(indices)), index_(index) {}

    constexpr reference operator*() const noexcept
    {
        return {*indices_, *data_};
    }

    constexpr CompressedRangeIterator &operator++() noexcept
    {
        ++indices_;
        ++data_;
        ++index_;
        return *this;
    }

    constexpr CompressedRangeIterator operator++(int) noexcept
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    friend constexpr bool operator==(
        CompressedRangeIterator const &lhs,
        CompressedRangeIterator const &rhs) noexcept
    {
        return lhs.index_ == rhs.index_;
    }

    friend constexpr bool operator!=(
        CompressedRangeIterator const &lhs,
        CompressedRangeIterator const &rhs) noexcept
    {
        return lhs.index_ != rhs.index_;
    }

    friend constexpr bool operator==(CompressedRangeIterator const& lhs, Sentinel const &rhs) noexcept
    {
        return lhs.index_ == rhs.end_index_;
    }

    friend constexpr bool operator!=(CompressedRangeIterator const& lhs, Sentinel const &rhs) noexcept
    {
        return lhs.index_ != rhs.end_index_;
    }

    static constexpr Sentinel end(difference_type end_index) noexcept
    {
        return {end_index};
    }
};

template <typename DataIter_, typename IndicesIter_>
class CompressedRange
{
    using DataTraits = std::iterator_traits<DataIter_>;
    using OffsetTraits = std::iterator_traits<IndicesIter_>;
    using difference_type = std::common_type_t<
        typename DataTraits::difference_type,
        typename OffsetTraits::difference_type>;

public:
    using iterator = CompressedRangeIterator<DataIter_, IndicesIter_>;

private:

    iterator begin_;
    typename iterator::Sentinel end_;

public:

    using const_iterator = iterator;

    constexpr CompressedRange(DataIter_ data, IndicesIter_ indices, difference_type count)
        : begin_(std::move(data), std::move(indices)), end_(count)
    {}

    constexpr auto begin() const noexcept
    {
        return begin_;
    }

    constexpr auto end() const noexcept
    {
        return end_;
    }
};

enum class CompressedDim
{
    Rows,
    Cols
};

} // namespace dtl


inline constexpr auto CompressedRow = dtl::CompressedDim::Rows;
inline constexpr auto CompressedCol = dtl::CompressedDim::Cols;



template<typename DataIter_, typename OffsetIter_, typename IndicesIter_, dtl::CompressedDim CompressedDim_=CompressedRow>
class CompressedMatrix {
    using DataTraits = std::iterator_traits<DataIter_>;
    using OffsetTraits = std::iterator_traits<OffsetIter_>;
    using IndicesTraits = std::iterator_traits<IndicesIter_>;

    /*
     * For the purposes of consistency, the "compressed" dimension refers to
     * the dimension for which offsets are provided. So for a compressed-row
     * matrix, we are provided with n_rows (+1) offsets into the column indices
     * and data ranges. For a compressed-column matrix, we are provide with
     * n_cols (+1) offsets into the row indices and data ranges. Thus the "inner"
     * dimension refers to the dimension that is given indices: for csr the
     * inner dim is columns, and for csc the inner dim is rows.
     */

public:
    using difference_type = std::common_type_t<
        typename DataTraits::difference_type,
        typename OffsetTraits::difference_type,
        typename IndicesTraits::difference_type>;

    using Scalar = typename DataTraits::value_type;
    using Offset = typename OffsetTraits::value_type;
    using Index = typename IndicesTraits::value_type;

    static constexpr auto compressed_dim = CompressedDim_;

private:
    DataIter_ data_; // size = n_non_zero_
    OffsetIter_ offsets_; // size = n_non_zero_
    IndicesIter_ indices_; // size = n_offsets_ + 1

    difference_type n_non_zero_;
    difference_type n_offsets_;
    difference_type inner_dim_;

public:

    constexpr explicit CompressedMatrix(
        DataIter_ data,
        IndicesIter_ indices,
        difference_type n_non_zero,
        OffsetIter_ offsets,
        difference_type n_offsets,
        difference_type inner_dim
    )
        : data_(data), offsets_(offsets), indices_(indices), n_non_zero_(n_non_zero), n_offsets_(n_offsets),
          inner_dim_(inner_dim) {
        // basic sanity test
        assert(n_non_zero_ <= n_offsets_ * inner_dim_);
    }

    using compressed_range = dtl::CompressedRange<DataIter_, OffsetIter_>;

    constexpr compressed_range in_dim(Index dim) const noexcept
    {
        auto dim_begin = offsets_[dim];
        auto dim_end = offsets_[dim + 1];
        return {
            data_ + dim_begin,
            indices_ + dim_begin,
            dim_end - dim_begin
        };
    }

    constexpr difference_type rows() const noexcept
    {
        if constexpr (CompressedDim_ == CompressedRow) {
            return n_offsets_;
        } else {
            return inner_dim_;
        }
    }

    constexpr difference_type cols() const noexcept
    {
        if constexpr (CompressedDim_ == CompressedRow) {
            return inner_dim_;
        } else {
            return n_offsets_;
        }
    }

    constexpr difference_type non_zeros() const noexcept
    {
        return n_non_zero_;
    }

    decltype(auto) at(difference_type row, difference_type col) const noexcept
    {
        const auto& zero = scalars::ScalarTraits<Scalar>::zero;
        auto compr_dim = (CompressedDim_ == CompressedRow) ? row : col;
        auto inner_dim = (CompressedDim_ == CompressedRow) ? col : row;

        auto dim_begin = offsets_[compr_dim];
        auto dim_end = offsets_[compr_dim + 1];

        if (dim_begin == dim_end) {
            return zero;
        }

        const auto inner_begin = indices_[dim_begin];
        const auto inner_end = indices_[dim_end];
        auto inner = std::find(inner_begin, inner_end, inner_dim);
        if (inner == inner_end) {
            return zero;
        }

        return data_[dim_begin + static_cast<difference_type>(inner - inner_begin)];
    }



};

} // namespace rpy::compute


#endif //ROUGHPY_COMPUTE_COMMON_SPARSE_MATRIX_HPP
