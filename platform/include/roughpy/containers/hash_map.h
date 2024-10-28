//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_HASH_MAP_H
#define ROUGHPY_PLATFORM_CONTAINERS_HASH_MAP_H

#include <functional>
#include <unordered_map>

#include <roughpy/core/types.h>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename Key,
        typename Value,
        typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key>,
        typename Alloc = mem::small::PoolAllocator<pair<const Key, Value>>>
using HashMap = std::unordered_map<Key, Value, Hash, KeyEqual, Alloc>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_HASH_MAP_H