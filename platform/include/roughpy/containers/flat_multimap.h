//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_FLAT_MULTIMAP_H
#define ROUGHPY_PLATFORM_CONTAINERS_FLAT_MULTIMAP_H

#include <functional>

#include <boost/container/flat_map.hpp>

#include "roughpy/platform/memory.h"
#include <roughpy/core/types.h>

#include "vector.h"

namespace rpy {

template <
        typename Key,
        typename Value,
        typename Compare = std::less<Key>,
        typename Alloc = mem::AlignedAllocator<pair<Key, Value>>>
using FlatMultiMap
        = boost::container::flat_multimap<Key, Value, Compare, Alloc>;

}// namespace rpy

#endif// ROUGHPY_PLATFORM_CONTAINERS_FLAT_MULTIMAP_H
