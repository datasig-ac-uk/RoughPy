
#include <algorithm>
#include <type_traits>

#include "roughpy_compute/common/basis.hpp"
#include "roughpy_compute/common/cache_array.hpp"
#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {

namespace dtl {

template <typename Int>
class BitMask
{
    using UInt = std::make_unsigned_t<Int>;
    UInt base_ = 0;

    constexpr BitMask() = default;

    constexpr explicit BitMask(int n_bits)
        : base_((UInt{1} << n_bits) - UInt{1})
    {}

    constexpr UInt operator[](int idx) const noexcept
    {
        assert(idx < 8 * sizeof(UInt));
        return (base_ >> idx) & UInt{1};
    }

    constexpr BitMask& operator++() noexcept
    {
        ++base_;
        return *this;
    }

    constexpr BitMask operator++(int) noexcept
    {
        BitMask tmp = *this;
        ++(*this);
        return tmp;
    }

    friend constexpr bool
    operator<(const BitMask& lhs, const BitMask& rhs) noexcept
    {
        return lhs.base_ < rhs.base_;
    }

    friend constexpr bool
    operator==(const BitMask& lhs, const BitMask& rhs) noexcept
    {
        return lhs.base_ == rhs.base_;
    }
};

}// namespace dtl

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
    using Scalar = typename DenseTensorView<OutIter>::Scalar;
    using Degree = typename DenseTensorView<OutIter>::Degree;
    using Index = typename DenseTensorView<OutIter>::Index;
    using Mask = dtl::BitMask<Index>;

    const Index width = out.width();
    const Index tile_size = width;

    CacheArray<int16_t, 32> letters(out.max_degree());
    CacheArray<Scalar, 8> tile(tile_size);

    out[0] += lhs[0] * rhs[0];
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
                out_level[i * width + j] += tile[j];
            }
        }

        out_size *= width;
    }
}

}// namespace v1
}// namespace rpy::compute::basic