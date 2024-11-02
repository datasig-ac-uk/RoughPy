//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_SET_H
#define ROUGHPY_PLATFORM_CONTAINERS_SET_H

#include <functional>
#include <set>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename T,
        typename Compare = std::less<T>,
        typename Alloc = std::allocator<T>>
using Set = std::set<T, Compare>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_SET_H
