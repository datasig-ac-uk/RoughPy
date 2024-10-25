//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_FLAT_SET_H
#define ROUGHPY_PLATFORM_CONTAINERS_FLAT_SET_H

#include <functional>

#include <boost/container/flat_set.hpp>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename T,
        typename Compare = std::less<T>,
        typename Allocator = mem::small::PoolAllocator<T>>
using FlatSet = boost::container::flat_set<T, Compare>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_FLAT_SET_H
