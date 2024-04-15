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
void preload_map(
        KeyScalarMap& mapped,
        IndexAndKeyRange&& range,
        const scalars::ScalarArray& scalars
)
{
    for (auto [i, k] : range) { mapped[k] = scalars[i]; }
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
        keys[count] = std::move(k);
        scalars[count] = v;
        ++count;
    }

    data.set_size(count);
}

}// namespace algebra
}// namespace rpy

#endif// SPARSE_HELPERS_H
