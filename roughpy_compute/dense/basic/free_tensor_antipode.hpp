#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP

#include <algorithm>

#include "roughpy_compute/common/cache_array.hpp"
#include "roughpy_compute/common/scalars.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {

struct BasicAntipodeConfig {
    static constexpr int32_t tile_letters = 1;
    static constexpr int32_t max_width = 5;
};

template <typename Index, typename Degree>
constexpr Index reverse_index(Index arg, Index width, Degree degree)
{
    Index result = 0;
    for (Degree i = 0; i < degree; ++i) {
        result *= width;
        result += arg % width;
        arg /= width;
    }
    return result;
}

struct DefaultSigner {
    bool is_odd = false;

    template <typename Degree>
    void for_degree(Degree d)
    {
        is_odd = (d % 2) != 0;
    }

    template <typename Scalar>
    constexpr Scalar operator()(Scalar arg) const noexcept
    {
        return (is_odd) ? -arg : arg;
    }
};

template <
        typename Context,
        typename OutIter,
        typename InIter,
        typename AntipodeConfig,
        typename Signer>
void ft_antipode(
        Context const& ctx,
        DenseTensorView<OutIter> out,
        DenseTensorView<InIter> arg,
        AntipodeConfig const& config,
        Signer&& signer
)
{
    using Index = typename DenseTensorView<OutIter>::Index;
    using Degree = typename DenseTensorView<OutIter>::Degree;

    auto const min_degree = std::max(arg.min_degree(), out.min_degree());
    auto const max_degree = std::min(arg.max_degree(), out.max_degree());
    Index const width = arg.width();

    // The unit element is always mapped to the unit element (without signing).
    if (min_degree == 0) {
        signer.for_degree(0);
        out[0] = signer(arg[0]);
    }

    // The level one elements are mapped to the corresponding level one
    // elements, with signing
    if (min_degree <= 1) {
        signer.for_degree(1);
        auto arg_level = arg.at_level(1);
        auto out_level = out.at_level(1);

        for (Index i = 0; i < arg_level.size(); ++i) {
            out_level[i] = signer(arg_level[i]);
        }
    }

    /*
     * For now, let's just write the untiled loop for the remainder of the
     * tensor. In the future, we can stop this loop at 2*tile_letters and
     * perform the rest using a tiled strategy which should dramatically improve
     * performance.
     */
    auto const start_degree = std::max(2, min_degree);
    for (Degree degree = start_degree; degree <= max_degree; ++degree) {
        signer.for_degree(degree);

        auto arg_level = arg.at_level(degree);
        auto out_level = out.at_level(degree);

        for (Index i = 0; i < arg_level.size(); ++i) {
            auto rev_idx = reverse_index(i, width, degree);
            out_level[rev_idx] = signer(arg_level[i]);
        }
    }
}

template <
        typename OutIter,
        typename InIter,
        typename AntipodeConfig,
        typename Signer>
void ft_antipode(
        DenseTensorView<OutIter> out,
        DenseTensorView<InIter> arg,
        AntipodeConfig const& config,
        Signer&& signer
)
{
    using Traits = scalars::Traits<typename DenseTensorView<OutIter>::Scalar>;

    return ft_antipode(
            Traits{},
            std::move(out),
            std::move(arg),
            config,
            std::forward<Signer>(signer)
    );
}

}// namespace v1
}// namespace rpy::compute::basic

#endif// ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP
