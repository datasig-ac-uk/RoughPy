//
// Created by sam on 26/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_SMALL_FLAT_MAP_H
#define ROUGHPY_PLATFORM_CONTAINERS_SMALL_FLAT_MAP_H

#include "flat_map.h"
#include "small_vector.h"

#include <roughpy/core/types.h>

namespace rpy {

template <
        typename Key,
        typename Value,
        dimn_t InlineCapacity,
        typename Compare = std::less<Key>,
        typename Alloc = mem::align::AlignedAllocator<Value>>
using SmallFlatMap = boost::container::flat_map < Key,
      Value, Compare, SmallVector<pair<const Key, Value>, InlineCapcity, Alloc>>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_SMALL_FLAT_MAP_H
