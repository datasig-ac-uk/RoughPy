#ifndef ROUGHPY_COMPUTE_COMMON_BASIS_HPP
#define ROUGHPY_COMPUTE_COMMON_BASIS_HPP

#include <algorithm>

namespace rpy::compute {

struct BasisBase {
    using Size = std::size_t;
    using Index = std::ptrdiff_t;
    using Degree = int32_t;

    Index const* degree_begin;// size depth + 2
    Degree width;
    Degree depth;

    Index size() const noexcept { return degree_begin[depth + 1]; }
};

struct TensorBasis : BasisBase {
    using Base = BasisBase;
    using typename Base::Degree;
    using typename Base::Size;

    [[nodiscard]]
    constexpr TensorBasis truncate(Degree new_depth) const noexcept
    {
        return {
                this->degree_begin,
                this->width,
                std::min(this->depth, new_depth),
        };
    }

    template <typename Letter>
    static void unpack_index_to_letters(
            Letter* letter_array,
            const Degree degree,
            Index index,
            const Index width
    ) noexcept
    {
        for (Degree d=0; d<degree; ++d) {
            letter_array[d] = static_cast<Letter>(index % width);
            index /= width;
        }
    }

    template <typename Letter, typename BitMask>
    static void pack_masked_index(Letter const* letters, Index degree, Index width, BitMask const& bitmask, Degree& lhs_deg, Index& lhs_idx, Degree& rhs_deg, Index& rhs_idx) noexcept
    {
        for (; degree >= 0; --degree) {
            if (bitmask[degree]) {
                ++lhs_deg;
                lhs_idx = lhs_idx * width + letters[degree];
            } else {
                ++rhs_deg;
                rhs_idx = rhs_idx * width + letters[degree];
            }
        }
    }
};

struct LieBasis : BasisBase {
    using Base = BasisBase;
    using typename Base::Degree;
    using typename Base::Index;

    Index const* data;

    [[nodiscard]]
    constexpr LieBasis truncate(Degree new_depth) const noexcept
    {
        return {
                {this->degree_begin,
                 this->width,
                 std::min(this->depth, new_depth)},
                this->data
        };
    }
};

constexpr typename BasisBase::Index data_size_to_degree(TensorBasis const& basis, typename BasisBase::Degree degree) noexcept
{
    if (degree > basis.depth) {
        degree = basis.depth;
    }
    return basis.degree_begin[degree + 1];
}

constexpr typename BasisBase::Index data_size_to_degree(LieBasis const& basis, typename BasisBase::Degree degree) noexcept
{
    if (degree > basis.depth) {
        degree = basis.depth;
    }
    return basis.degree_begin[degree + 1] - 1;
}


}// namespace rpy::compute

#endif// ROUGHPY_COMPUTE_COMMON_BASIS_HPP
