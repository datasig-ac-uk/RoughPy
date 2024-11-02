//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_FLAT_MAP_H
#define ROUGHPY_PLATFORM_CONTAINERS_FLAT_MAP_H

#include <functional>

#include <boost/container/flat_map.hpp>

#include <roughpy/core/types.h>
#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename Key,
        typename Value,
        typename Compare = std::less<Key>,
        typename Alloc = std::allocator<pair<Key, Value>>>
using FlatMap = boost::container::flat_map<Key, Value, Compare, Alloc>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_FLAT_MAP_H
