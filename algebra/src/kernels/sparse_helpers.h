//
// Created by sam on 4/15/24.
//

#ifndef SPARSE_HELPERS_H
#define SPARSE_HELPERS_H

#include "common.h"
#include "generic_kernel.h"

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/ranges.h>

namespace rpy {
namespace algebra {

using KeyScalarMap = containers::
        FlatHashMap<BasisKey, scalars::Scalar, KeyHash, KeyEquals>;

template <typename IndexAndKeyRange, typename F>
void write_with_sparse(
        KeyScalarMap& mapped,
        IndexAndKeyRange&& range,
        const scalars::ScalarArray& scalars,
        F&& func
)
{
    for (auto [i, k] : range) { func(mapped[k], scalars[i]); }
}

template <typename IndexAndKeyRange>
KeyScalarMap preload_map(
        const Basis* basis,
        IndexAndKeyRange&& range,
        const scalars::ScalarArray& scalars
)
{
    const auto scalar_view = scalars.view();
    KeyScalarMap mapped(0, KeyHash{basis}, KeyEquals{basis});
    for (auto [i, k] : range) { mapped[k] = scalar_view[i]; }
    return mapped;
}

inline bool filter_pairs(typename KeyScalarMap::reference val) noexcept
{
    return val.second.is_zero();
}

inline void write_sparse_result(VectorData& data, KeyScalarMap& mapped)
{
    const auto mapped_size = mapped.size();
    if (data.capacity() < mapped_size) { data.reserve(mapped_size); }

    auto keys = data.mut_keys().mut_view();
    auto scalars = data.mut_scalars().mut_view();

    dimn_t count = 0;
    for (auto&& [k, v] : mapped | views::filter(filter_pairs) | views::move) {
        keys[count] = k;
        scalars[count] = v;
        ++count;
    }

    data.set_size(count);
}


template <typename IKR, typename F>
void binary_operation_into_map_left(
    KeyScalarMap& mapped,
    IKR&& ikr,
    const scalars::ScalarArray& left,
    F&& func,
    const scalars::Scalar& zero
    )
{
    for (auto [i, k] : ikr) {
        func(mapped[k], left[i], zero);
    }
}

template <typename IKR, typename F>
void binary_operation_into_map_right(
    KeyScalarMap& mapped,
    IKR&& ikr,
    const scalars::ScalarArray& right,
    F&& func,
    const scalars::Scalar& zero
    )
{
    for (auto [i, k] : ikr) {
        func(mapped[k], zero, right[i]);
    }
}

template <typename IDK1, typename IDK2, typename F>
void binary_operation_into_map(
        KeyScalarMap& mapped,
        IDK1&& idk_left,
        const scalars::ScalarArray& left,
        IDK2&& idk_right,
        const scalars::ScalarArray& right,
        F&& func
)
{
    const scalars::Scalar zero(left.type());
    binary_operation_into_map_left(mapped, idk_left, left, func, zero);
    binary_operation_into_map_right(mapped, idk_right, right, func, zero);
}


}// namespace algebra
}// namespace rpy

#endif// SPARSE_HELPERS_H
