//
// Created by sam on 3/30/24.
//

#ifndef ROUGHPY_CORE_CONTAINERS_UNORDERED_SET_H
#define ROUGHPY_CORE_CONTAINERS_UNORDERED_SET_H

#include <roughpy/core/alloc.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/types.h>

#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#include <boost/unordered/unordered_set.hpp>

#include <functional>

namespace rpy {
namespace containers {

template <
        typename T,
        typename H = Hash<T>,
        typename P = std::equal_to<T>,
        typename A = Allocator<T>>
using UnorderedSet = boost::unordered_set<T, H, P, A>;

template <
        typename T,
        typename H = Hash<T>,
        typename P = std::equal_to<T>,
        typename A = Allocator<T>>
using FlatHashSet = boost::unordered::unordered_flat_set<T, H, P, A>;

template <
        typename T,
        typename H = Hash<T>,
        typename P = std::equal_to<T>,
        typename A = Allocator<T>>
using NodeHashSet = boost::unordered::unordered_node_set<T, H, P, A>;

}// namespace containers
}// namespace rpy

#endif// ROUGHPY_CORE_CONTAINERS_UNORDERED_SET_H
