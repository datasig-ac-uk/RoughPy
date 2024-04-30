//
// Created by sam on 4/24/24.
//

#ifndef MULTIPLICATION_IMPL_H
#define MULTIPLICATION_IMPL_H

#include "sparse_helpers.h"

#include <roughpy/core/ranges.h>
namespace rpy {
namespace algebra {
namespace dtl {

template <
        typename Mapped,
        typename KeyFunc,
        typename ScalarFunc,
        typename LeftRng,
        typename RightRng>
void square_generic_multiplication(
        Mapped& out_map,
        KeyFunc&& key_fn,
        ScalarFunc&& scalar_fn,
        LeftRng&& left,
        const scalars::ScalarArray& left_data,
        RightRng&& right,
        const scalars::ScalarArray& right_data
)
{

    for (const auto& [li, lkey] : left) {
        for (const auto& [ri, rkey] : right) {
            for (const auto& [okey, oscal] : key_fn(lkey, rkey)) {
                auto out_scal = out_map[okey];
                scalar_fn(out_scal, oscal * left_data[li], right_data[ri]);
            }
        }
    }
}

template <
        typename Mapped,
        typename KeyFunc,
        typename ScalarFunc,
        typename LeftRng,
        typename RightRng>
void triangular_unorderded_generic_multiplication(
        Mapped& out_map,
        KeyFunc&& key_fn,
        ScalarFunc&& scalar_fn,
        const Basis* basis,
        LeftRng&& left,
        const scalars::ScalarArray& left_data,
        RightRng&& right,
        const scalars::ScalarArray& right_data,
        deg_t max_degree
)
{
    containers::Vec<containers::Vec<pair<BasisKey, scalars::Scalar>>>
            level_data(max_degree + 1);
    for (auto&& [key, val] : right) {
        auto degree = basis->degree(key);
        if (degree <= max_degree) {
            level_data[degree].emplace_back(
                    std::forward<decltype(key)>(key),
                    val
            );
        }
    }

    for (const auto& [lkey, li] : left) {
        const auto ldegree = basis->degree(lkey);
        if (ldegree > max_degree) { continue; }

        const auto lscal = left_data[li];

        for (const auto& [rkey, ri] : level_data[max_degree - ldegree]) {
            for (const auto& [okey, oscal] : key_fn(lkey, rkey)) {
                scalar_fn(out_map[okey], oscal * lscal, right_data[ri]);
            }
        }
    }
}

template <
        typename Mapped,
        typename KeyFunc,
        typename ScalarFunc,
        typename LeftRng,
        typename RightRng>
void triangular_ordered_generic_multiplication_right_dense(
        Mapped& mapped,
        KeyFunc&& key_fn,
        ScalarFunc&& scalar_fn,
        const Basis* basis,
        LeftRng&& left,
        const scalars::ScalarArray& left_data,
        RightRng&& right,
        const scalars::ScalarArray& right_data,
        deg_t max_degree
)
{
    containers::Vec<decltype(right | views::slice(0, 0))> levels;
    levels.reserve(max_degree + 1);
    dimn_t dimension = 0;
    for (deg_t d = 0; d <= max_degree; ++d) {
        const auto next_dim = basis->dimension_to_degree(d);
        levels.emplace_back(right | views::slice(dimension, next_dim));
        dimension = next_dim;
    }

    for (const auto& [li, lkey] : left) {
        const auto ldegree = basis->degree(lkey);
        const auto rdegree = (ldegree <= max_degree) ? max_degree - ldegree : 0;
        const auto lscal = left_data[li];
        for (const auto& [ri, rkey] : levels[rdegree]) {
            for (const auto& [okey, oscal] : key_fn(lkey, rkey)) {
                auto out_scal = mapped[okey];
                scalar_fn(out_scal, oscal * lscal, right_data[ri]);
            }
        }
    }
}

template <
        typename Mapped,
        typename KeyFunc,
        typename ScalarFunc,
        typename LeftRng,
        typename RightRng>
void triangular_ordered_generic_multiplication_right_sparse(
        Mapped& mapped,
        KeyFunc&& key_fn,
        ScalarFunc&& scalar_fn,
        const Basis* basis,
        LeftRng&& left,
        const scalars::ScalarArray& left_data,
        RightRng&& right,
        const scalars::ScalarArray& right_data,
        deg_t max_degree
)
{
    containers::Vec<pair<dimn_t, dimn_t>> levels;
    levels.reserve(max_degree + 1);
    auto first = ranges::begin(right);
    deg_t last_degree = 0;
    const auto rbegin = ranges::begin(right);
    for (auto it = ranges::begin(right); it != ranges::end(right); ++it) {
        const auto degree = basis->degree(std::get<1>(*it));
        if (degree > max_degree) { break; }
        for (; last_degree < degree; ++last_degree) {
            levels.emplace_back(
                    ranges::distance(rbegin, first),
                    ranges::distance(rbegin, it)
            );
            first = it;
        }
    }
    {
        const auto back = levels.back().second;
        for (; last_degree <= max_degree; ++last_degree) {
            levels.emplace_back(back, back);
        }
    }

    for (const auto& [li, lkey] : left) {
        const auto ldegree = basis->degree(lkey);
        const auto rdegree = (ldegree <= max_degree) ? max_degree - ldegree : 0;
        const auto lscal = left_data[li];
        const auto [bi, ei] = levels[rdegree];
        for (auto rit=rbegin+bi; rit != rbegin+ei; ++rit) {
            auto&& [ri, rkey] = *rit;
            const auto rscal = right_data[ri];
            for (const auto& [okey, oscal] : key_fn(lkey, rkey)) {
                auto out_scal = mapped[okey];
                scalar_fn(out_scal, oscal * lscal, rscal);
            }
        }
    }
}

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// MULTIPLICATION_IMPL_H
