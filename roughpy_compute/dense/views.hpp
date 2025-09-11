#ifndef ROUGHPY_COMPUTE_DENSE_VIEWS_HPP
#define ROUGHPY_COMPUTE_DENSE_VIEWS_HPP

#include <cassert>
#include <type_traits>
#include <utility>

#include "roughpy_compute/common/basis.hpp"
#include "roughpy_compute/common/iterators.hpp"

namespace rpy::compute {
inline namespace v1 {
namespace dtl {
/**
 * @brief Dense index-value iterator for dense vector fragments
 *
 * This is a little overengineered. Since dense vectors require random access
 * iterators, the concept checking (using sfinae, because C++17) is not
 * necessary. However, this class might be useful elsewhere so it's designed
 * with this in mind.
 */
template<typename Iter_>
class DenseIndexValueIterator {
    using Traits = std::iterator_traits<Iter_>;

public:
    using Scalar = typename Traits::value_type;
    using ScalarRef = typename Traits::reference;

    using difference_type = typename Traits::difference_type;

    using value_type = std::pair<const difference_type, Scalar>;
    using reference = std::pair<const difference_type, ScalarRef>;

private:
    Iter_ iter_;
    difference_type index_;

public:
    constexpr explicit DenseIndexValueIterator(Iter_ iter, difference_type index = 0) noexcept
        : iter_(iter), index_(index) {
    }

    constexpr Iter_ const &base() const noexcept { return iter_; }
    constexpr difference_type const &index() const noexcept { return index_; }

    constexpr reference operator*() const noexcept {
        return reference{index_, *iter_};
    }

    constexpr DenseIndexValueIterator &operator++() noexcept {
        ++iter_;
        ++index_;
        return *this;
    }

    constexpr DenseIndexValueIterator operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    friend constexpr bool operator==(
        DenseIndexValueIterator const &lhs,
        DenseIndexValueIterator const &rhs) noexcept {
        return lhs.iter_ == rhs.iter_;
    }

    friend constexpr bool operator!=(
        DenseIndexValueIterator const &lhs,
        DenseIndexValueIterator const &rhs
    ) noexcept {
        return lhs.iter_ != rhs.iter_;
    }


    template<typename I=Iter_>
    constexpr
    std::enable_if_t<is_bidirectional_iterator_v<I>,
        DenseIndexValueIterator &>
    operator--() noexcept {
        --iter_;
        --index_;
        return *this;
    }

    template<typename I=Iter_>
    constexpr std::enable_if_t<is_bidirectional_iterator_v<I>, DenseIndexValueIterator>
    operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    template<typename I=Iter_>
    constexpr std::enable_if_t<is_random_access_v<I>, DenseIndexValueIterator>
    operator+=(difference_type n) noexcept {
        iter_ += n;
        index_ += n;
        return *this;
    }

    template<typename I=Iter_>
    constexpr std::enable_if_t<is_random_access_v<I>, DenseIndexValueIterator>
    operator-=(difference_type n) noexcept {
        iter_ -= n;
        index_ -= n;
        return *this;
    }

    template<typename I=Iter_>
    constexpr std::enable_if_t<is_random_access_v<I>, reference>
    operator[](difference_type n) noexcept {
        return reference{index_ + n, *(iter_ + n)};
    }
};

template<typename Iter_, typename Difference_>
constexpr
std::enable_if_t<is_random_access_v<Iter_>, DenseIndexValueIterator<Iter_> >
operator+(DenseIndexValueIterator<Iter_> const &iter, Difference_ n) noexcept {
    using Index = typename std::iterator_traits<Iter_>::difference_type;
    const auto offset = static_cast<Index>(n);
    return DenseIndexValueIterator<Iter_>{iter.base() + offset, iter.index() + offset};
}

template<typename Iter_, typename Difference_>
constexpr
std::enable_if_t<is_random_access_v<Iter_>, DenseIndexValueIterator<Iter_> >
operator+(Difference_ n, DenseIndexValueIterator<Iter_> const &iter) noexcept {
    using Index = typename std::iterator_traits<Iter_>::difference_type;
    const auto offset = static_cast<Index>(n);
    return DenseIndexValueIterator<Iter_>{iter.base() + offset, iter.index() + offset};
}


template<typename Iter_, typename Difference_>
constexpr
std::enable_if_t<is_random_access_v<Iter_>, DenseIndexValueIterator<Iter_> >
operator-(DenseIndexValueIterator<Iter_> const &iter, Difference_ n) noexcept {
    using Index = typename std::iterator_traits<Iter_>::difference_type;
    const auto offset = static_cast<Index>(n);
    return DenseIndexValueIterator<Iter_>{iter.base() + offset, iter.index() + offset};
}

template<typename Iter_>
constexpr std::enable_if_t<is_random_access_v<Iter_>, bool>
operator<(DenseIndexValueIterator<Iter_> const &lhs, DenseIndexValueIterator<Iter_> const &rhs) noexcept {
    return lhs.base() < rhs.base();
}

template<typename Iter_>
constexpr std::enable_if_t<is_random_access_v<Iter_>, bool>
operator<=(DenseIndexValueIterator<Iter_> const &lhs, DenseIndexValueIterator<Iter_> const &rhs) noexcept {
    return lhs.base() <= rhs.base();
}

template<typename Iter_>
constexpr std::enable_if_t<is_random_access_v<Iter_>, bool>
operator>(DenseIndexValueIterator<Iter_> const &lhs, DenseIndexValueIterator<Iter_> const &rhs) noexcept {
    return lhs.base() > rhs.base();
}

template<typename Iter_>
constexpr std::enable_if_t<is_random_access_v<Iter_>, bool>
operator>=(DenseIndexValueIterator<Iter_> const &lhs, DenseIndexValueIterator<Iter_> const &rhs) noexcept {
    return lhs.base() >= rhs.base();
}
};


template<typename Iter_>
class StridedDenseIterator {
    using Traits = std::iterator_traits<Iter_>;
    static_assert(
        is_random_access_v<Iter_>,
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
        : base_(base), stride_(stride) {
    }

    constexpr StridedDenseIterator(Iter_ base,
                                   difference_type stride,
                                   difference_type offset) noexcept
        : base_(base + offset), stride_(stride) {
    }


    [[nodiscard]]
    constexpr reference operator*() const noexcept { return *base_; }

    [[nodiscard]]
    constexpr pointer operator->() const noexcept { return base_; }

    [[nodiscard]]
    constexpr reference operator[](difference_type n) const noexcept {
        return base_[n * stride_];
    }

    constexpr StridedDenseIterator &operator++() noexcept {
        base_ += stride_;
        return *this;
    }

    [[nodiscard]]
    constexpr StridedDenseIterator operator++(int) noexcept {
        auto tmp = *this;
        base_ += stride_;
        return tmp;
    }

    constexpr StridedDenseIterator &operator--() noexcept {
        base_ -= stride_;
        return *this;
    }

    [[nodiscard]]
    constexpr StridedDenseIterator operator--(int) noexcept {
        auto tmp = *this;
        base_ -= stride_;
        return tmp;
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator+(
        StridedDenseIterator const &iter,
        difference_type offset) noexcept {
        return {iter.base_, iter.stride_, offset};
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator+(
        difference_type offset,
        StridedDenseIterator const &iter) noexcept {
        return {iter.base_, iter.stride_, offset};
    }

    [[nodiscard]]
    friend constexpr StridedDenseIterator operator-(
        StridedDenseIterator const &iter,
        difference_type offset) noexcept {
        return {iter.base_, iter.stride_, -offset};
    }

    friend constexpr difference_type operator-(StridedDenseIterator const &lhs,
                                               StridedDenseIterator const &rhs) {
        return static_cast<difference_type>(lhs.base_ - rhs.base_) / static_cast
               <difference_type>(rhs.stride_);
    }

    friend constexpr bool operator==(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ == rhs.base_;
    }

    friend constexpr bool operator!=(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ != rhs.base_;
    }

    friend constexpr bool operator<(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ < rhs.base_;
    }

    friend constexpr bool operator<=(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ <= rhs.base_;
    }

    friend constexpr bool operator>(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ > rhs.base_;
    }

    friend constexpr bool operator>=(
        StridedDenseIterator const &lhs,
        StridedDenseIterator const &rhs) noexcept {
        return lhs.base_ >= rhs.base_;
    }
};


template<typename Iter_>
class DenseVectorFragment {
    using Traits = std::iterator_traits<Iter_>;

    Iter_ base_;
    typename Traits::difference_type size_;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using difference_type = typename Traits::difference_type;

    using iterator = dtl::DenseIndexValueIterator<Iter_>;
    using const_iterator = iterator;

    using Scalar = value_type;

    constexpr DenseVectorFragment(Iter_ begin, difference_type size) noexcept
        : base_(begin), size_(size) {
    }

    template<typename Index_>
    constexpr reference operator[](Index_ index) noexcept {
        assert(static_cast<difference_type>(index) < size_);
        return base_[index];
    }

    constexpr difference_type size() const noexcept { return size_; }


    [[nodiscard]]
    constexpr const_iterator begin() const noexcept {
        return const_iterator{base_, 0};
    }

    [[nodiscard]]
    constexpr const_iterator end() const noexcept {
        return const_iterator{base_, size()};
    }
};


template<typename Iter_, typename Basis_>
class DenseVectorView {
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
          max_degree_{max_degree} {
        if (max_degree_ == -1) { max_degree_ = basis_.depth; }
    }


    [[nodiscard]]
    constexpr Iter_ data() const noexcept { return data_; }

    [[nodiscard]]
    constexpr Basis const &basis() const noexcept { return basis_; }


    [[nodiscard]]
    constexpr Degree width() const noexcept { return basis_.width; }

    [[nodiscard]]
    constexpr Degree depth() const noexcept { return max_degree_; }

    [[nodiscard]]
    constexpr Index size() const noexcept {
        return basis_.degree_begin[max_degree_ + 1] - basis_.degree_begin[
                   min_degree_];
    }

    [[nodiscard]]
    constexpr Degree max_degree() const noexcept { return max_degree_; }

    [[nodiscard]]
    constexpr Degree min_degree() const noexcept { return min_degree_; }


    [[nodiscard]]
    constexpr DenseVectorFragment<Iter_> at_level(Degree degree) const noexcept {
        assert(min_degree_ <= degree && degree <= max_degree_);
        auto start_of_degree = basis_.degree_begin[degree];
        return {
            data_ + start_of_degree,
            (basis_.degree_begin[degree + 1] -
             start_of_degree)
        };
    }


    [[nodiscard]]
    constexpr reference operator[](Index i) noexcept { return data_[i]; }
};


template<typename Iter_>
class DenseTensorView : public DenseVectorView<Iter_, TensorBasis> {
public:
    using Base = DenseVectorView<Iter_, TensorBasis>;
    using typename Base::Degree;

    using Base::Base;

    constexpr DenseTensorView truncate(Degree new_max_degree,
                                       Degree new_min_degree = 0) const noexcept {
        return {
                this->data(),
                this->basis(),
                std::max(this->min_degree(), new_min_degree),
                std::min(this->max_degree(), new_max_degree)
        };
    }
};


template<typename Iter_>
class DenseLieView : public DenseVectorView<Iter_, LieBasis> {
public:
    using Base = DenseVectorView<Iter_, LieBasis>;
    using typename Base::Degree;

    using Base::Base;

    [[nodiscard]]
    constexpr DenseLieView truncate(Degree new_max_degree,
                                    Degree new_min_degree = 0) const noexcept {
        return {
            this->data(),
            this->basis(),
            std::max(this->min_degree(), new_min_degree),
            std::min(this->max_degree(), new_max_degree)
        };
    }
};
}
} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE_DENSE_VIEWS_HPP