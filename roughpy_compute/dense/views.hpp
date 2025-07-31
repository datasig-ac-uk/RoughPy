#ifndef ROUGHPY_COMPUTE_DENSE_VIEWS_HPP
#define ROUGHPY_COMPUTE_DENSE_VIEWS_HPP

#include "tensor_basis.h"

#include <cassert>
#include <iterator>
#include <type_traits>

#include "roughpy_compute/common/architecture.hpp"
#include "roughpy_compute/common/basis.hpp"

namespace rpy::compute {
inline namespace v1 {


template <typename Iter_>
class DenseVectorFragment {
    using Traits = std::iterator_traits<Iter_>;
    using size_type = std::size_t;

    Iter_ base_;
    size_type size_;

public:
    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;
    using iterator = Iter_;

    constexpr DenseVectorFragment(Iter_ begin, std::size_t size) noexcept
        : base_(begin), size_(size) {}

    template <typename Index_>
    constexpr reference operator[](Index_ index) noexcept {
        assert(static_cast<std::size_t>(index) < size_);
        return base_[index];
    }

    constexpr size_type size() const noexcept { return size_; }
};



template <typename Iter_, typename Basis_>
class DenseVectorView
{
    Basis_ basis_
    Iter_ data_;

    using Traits = std::iterator_traits<Iter_>;

    static_assert(
        std::is_base_of_v<std::random_access_iterator_tag, typename
            Traits::iterator_category> || std::is_same_v<
            std::random_access_iterator_tag, typename
            Traits::iterator_category>,
        "iterator must be random access");

    Degree min_degree_;
    Degree max_degree_;

public:
    using Basis = Basis_;
    using Architecture = typename Basis::Architecture;

    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;

    using value_type = typename Traits::value_type;
    using reference = typename Traits::reference;


    constexpr DenseVectorView(Iter_ data, Basis basis, Degree min_degree=0, Degree max_degree=-1)
        : Basis_(std::move(basis)), data_(data), min_degree_{min_degree}, max_degree_{max_degree_}
    {
        if (max_degree_ == -1) {
            max_degree_ = basis_.depth;
        }

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
    constexpr Size size() const noexcept { return basis_.size(); }

    [[nodiscard]]
    constexpr Degree max_degree() const noexcept { return max_degree_; }
    [[nodiscard]]
    constexpr Degree min_degree() const noexcept { return min_degree_; }



    [[nodiscard]]
    constexpr DenseVectorFragment at_level(Degree degree) const noexcept {
        assert(min_degree_ <= degree && degree <= max_degree_);
        auto start_of_degree = basis_.degree_begin[degree]
        return { data_ + start_of_degree,
                 static_cast<std::size_t>(basis_.begin[degree + 1] - start_of_degree)
        };
    }


    [[nodiscard]]
    constexpr reference operator[](Index i) noexcept {
        return data_[i];
    }

};



template <typename Iter_>
class DenseTensorView : public DenseTensorView <Iter_, TensorBasis<>>
{
    using Base = DenseTensorView <Iter_, TensorBasis<>>;
    using typename Base::Degree;




    constexpr DenseTensorView truncate(Degree new_max_degree, Degree new_min_degree=0) const noexcept
    {
        return {
            this->data(),
            this->basis(),
            std::max(min_degree_, new_min_degree),
            std::min(max_degree_, new_max_degree)
        };
    }

}



template <typename Iter_>
class DenseLieView : public DenseVectorView <Iter_, LieBasis<>>
{
    using Base = DenseVectorView <Iter_, LieBasis<>>;
    using typename Base::Degree;

    [[nodiscard]]
    constexpr DenseLieView truncate(Degree new_max_degree, Degree new_min_degree=0) const noexcept
    {
        return {
            this->data(),
            this->basis(),
            std::max(min_degree_, new_min_degree),
            std::min(max_degree_, new_max_degree)
        };
    }
};





}}// namespace rpy::compute

#endif //ROUGHPY_COMPUTE_DENSE_VIEWS_HPP