//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_SMALL_VECTOR_H
#define ROUGHPY_PLATFORM_CONTAINERS_SMALL_VECTOR_H

#include <boost/container/small_vector.hpp>

#include <roughpy/core/types.h>

#include "roughpy/platform/memory.h"

namespace rpy {

template <
        typename T,
        dimn_t InlineSize,
        typename Allocator = mem::AlignedAllocator<T>>
using SmallVector = boost::container::small_vector<T, InlineSize, Allocator>;

}

#endif// ROUGHPY_PLATFORM_CONTAINERS_SMALL_VECTOR_H
