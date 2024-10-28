//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_MAP_H
#define ROUGHPY_PLATFORM_CONTAINERS_MAP_H

#include <functional>
#include <map>

#include <roughpy/core/types.h>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename Key,
        typename Value,
        typename Compare = std::less<Key>,
        typename Alloc = mem::small::PoolAllocator<pair<const Key, Value>>>
using Map = std::map<Key, Value, Compare, Alloc>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_MAP_H
