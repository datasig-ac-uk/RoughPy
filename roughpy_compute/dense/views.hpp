#ifndef ROUGHPY_COMPUTE_DENSE_VIEWS_HPP
#define ROUGHPY_COMPUTE_DENSE_VIEWS_HPP

#include <cassert>
#include <iterator>
#include <type_traits>

#include "roughpy_compute/common/architecture.hpp"
#include "roughpy_compute/common/basis.hpp"

namespace rpy::compute {
inline namespace v1 {

namespace dtl {

template <typename Iter_>
inline constexpr bool is_random_access_v = std::is_base_of_v<
    std::random_access_iterator_tag,
    typename std::iterator_traits<Iter_>::iterator_category>;

};


template <typename Iter_>
class StridedDenseIterator
{
    using Traits = std::iterator_traits<Iter_>;
    static_assert(
        dtl::is_random_access_v<Iter_>,
        "base iterator must support random access"
    );

    Iter_ base_;
    typename Traits::difference_type stride_ = 0;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;

    /*
     * The striding means that in general we cannot support contiguous
     * iteration even if the underlying iterator does.
     */
    using iterator_category = std::random_access_iterator_tag;

    constexpr StridedDenseIterator(Iter_ base, difference_type stride) noexcept
        : base_(base), stride_(stride) {}

    constexpr StridedDenseIterator(Iter_ base,
                                   difference_type stride,
                                   difference_type offset) noexcept
        : base_(base + offset), stride_(stride) {}


    [[nodiscard]]
    constexpr reference operator*() const noexcept { return *base_; }

    [[nodiscard]]
    constexpr pointer operator->() const noexcept { return base_; }

    [[nodiscard]]
    constexpr reference operator[](difference_type n) const noexcept
    {
        return base_[n * stride_];
    }

    constexpr StridedDenseIterator& operator++() noexcept
    {
        base_ += stride_;
        return *this;
    }

    [[nodiscard]]
    constexpr StridedDenseIterator operator++(int) noexcept
    {
        auto tmp = *this;
        base_ += stride_;
        return tmp;
    }

    constexpr StridedDenseIterator& operator--() noexcept
    {
        base_ -= stride_;
        return *this;
    }

    [[nodiscard]]
    constexpr StridedDenseIterator operator--(int) noexcept
    {
        auto tmp = *this;
        base_ -= stride_;
        return tmp;
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator+(
        StridedDenseIterator const& iter,
        difference_type offset) noexcept
    {
        return {iter.base_, iter.stride_, offset};
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator+(
        difference_type offset,
        StridedDenseIterator const& iter) noexcept
    {
        return {iter.base_, iter.stride_, offset};
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator-(
        StridedDenseIterator const& iter,
        difference_type offset) noexcept
    {
        return {iter.base_, iter.stride_, -offset};
    }

    friend constexpr difference_type operator-(StridedDenseIterator const& lhs,
                                               StridedDenseIterator const& rhs)
    {
        return static_cast<difference_type>(lhs.base_ - rhs.base_) / static_cast
                <difference_type>(rhs.stride_);
    }

    friend constexpr bool operator==(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ == rhs.base_;
    }

    friend constexpr bool operator!=(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ != rhs.base_;
    }

    friend constexpr bool operator<(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ < rhs.base_;
    }

    friend constexpr bool operator<=(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ <= rhs.base_;
    }

    friend constexpr bool operator>(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ > rhs.base_;
    }

    friend constexpr bool operator>=(
        StridedDenseIterator const& lhs,
        StridedDenseIterator const& rhs) noexcept
    {
        return lhs.base_ >= rhs.base_;
    }

};


template <typename Iter_>
class DenseVectorFragment
{
    using Traits = std::iterator_traits<Iter_>;

    Iter_ base_;
    typename Traits::difference_type size_;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using difference_type = typename Traits::difference_type;
    using iterator = Iter_;

    using Scalar = value_type;

    constexpr DenseVectorFragment(Iter_ begin, difference_type size) noexcept
        : base_(begin), size_(size) {}

    template <typename Index_>
    constexpr reference operator[](Index_ index) noexcept
    {
        assert(static_cast<difference_type>(index) < size_);
        return base_[index];
    }

    constexpr difference_type size() const noexcept { return size_; }
};


template <typename Iter_, typename Basis_>
class DenseVectorView
{
    Basis_ basis_;
    Iter_ data_;

    using Traits = std::iterator_traits<Iter_>;

    static_assert(
        std::is_base_of_v<std::random_access_iterator_tag, typename
            Traits::iterator_category> || std::is_same_v<
            std::random_access_iterator_tag, typename
            Traits::iterator_category>,
        "iterator must be random access");

    typename Basis_::Degree min_degree_;
    typename Basis_::Degree max_degree_;

public:
    using Basis = Basis_;

    using Index = typename Traits::difference_type;
    using Degree = typename Basis::Degree;

    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;

    using Scalar = value_type;


    constexpr DenseVectorView(Iter_ data,
                              Basis basis,
                              Degree min_degree = 0,
                              Degree max_degree = -1)
        : basis_(std::move(basis)), data_(data), min_degree_{min_degree},
          max_degree_{max_degree}
    {
        if (max_degree_ == -1) { max_degree_ = basis_.depth; }
    }


    [[nodiscard]]
    constexpr Iter_ data() const noexcept { return data_; }

    [[nodiscard]]
    constexpr Basis const& basis() const noexcept { return basis_; }


    [[nodiscard]]
    constexpr Degree width() const noexcept { return basis_.width; }

    [[nodiscard]]
    constexpr Degree depth() const noexcept { return max_degree_; }

    [[nodiscard]]
    constexpr Index size() const noexcept
    {
        return basis_.degree_begin[max_degree_ + 1] - basis_.degree_begin[
            min_degree_];
    }

    [[nodiscard]]
    constexpr Degree max_degree() const noexcept { return max_degree_; }

    [[nodiscard]]
    constexpr Degree min_degree() const noexcept { return min_degree_; }

    [[nodiscard]]
    constexpr DenseVectorFragment<Iter_> at_level(Degree degree) const noexcept
    {
        assert(min_degree_ <= degree && degree <= max_degree_);
        auto start_of_degree = basis_.degree_begin[degree];
        return {data_ + start_of_degree,
                (basis_.degree_begin[degree + 1] -
                    start_of_degree)
        };
    }

    [[nodiscard]]
    constexpr DenseVectorFragment<Iter_> level_range(Degree degree_begin,
        Degree degree_end) const noexcept
    {
        auto begin_idx = basis_.degree_begin[degree_begin];
        auto end_idx = basis_.degree_begin[degree_end+1];
        return {data_ + begin_idx, end_idx - begin_idx};
    }

    [[nodiscard]]
    constexpr reference operator[](Index i) noexcept { return data_[i]; }

};


template <typename Iter_>
class DenseTensorView : public DenseVectorView<Iter_, TensorBasis>
{
public:
    using Base = DenseVectorView<Iter_, TensorBasis>;
    using typename Base::Degree;

    using Base::Base;

    constexpr DenseTensorView truncate(Degree new_max_degree,
                                       Degree new_min_degree = 0) const noexcept
    {

        return {
                this->data(),
                this->basis(),
                std::max(this->min_degree(), new_min_degree),
                std::min(this->max_degree(), new_max_degree)
        };
    }

};


template <typename Iter_>
class DenseLieView : public DenseVectorView<Iter_, LieBasis>
{
public:
    using Base = DenseVectorView<Iter_, LieBasis>;
    using typename Base::Degree;

    using Base::Base;

    [[nodiscard]]
    constexpr DenseLieView truncate(Degree new_max_degree,
                                    Degree new_min_degree = 0) const noexcept
    {
        return {
                this->data(),
                this->basis(),
                std::max(this->min_degree(), new_min_degree),
                std::min(this->max_degree(), new_max_degree)
        };
    }
};


}
}// namespace rpy::compute

#endif //ROUGHPY_COMPUTE_DENSE_VIEWS_HPP