//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_HASH_SET_H
#define ROUGHPY_PLATFORM_CONTAINERS_HASH_SET_H

#include <functional>
#include <unordered_set>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename T,
        typename Hash = std::hash<T>,
        typename KeyEqual = std::equal_to<T>,
        typename Alloc = std::allocator<T>>
using HashSet = std::unordered_set<T, Hash, KeyEqual, Alloc>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_HASH_SET_H
