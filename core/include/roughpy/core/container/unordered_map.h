//
// Created by sam on 3/25/24.
//

#ifndef ROUHGPY_CORE_CONTAINER_UNORDERD_MAP_H
#define ROUHGPY_CORE_CONTAINER_UNORDERD_MAP_H

#include <roughpy/core/alloc.h>
#include <roughpy/core/hash.h>
#include <roughpy/core/types.h>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_map.hpp>
#include <boost/unordered/unordered_node_map.hpp>

namespace rpy {
namespace containers {

template <
        typename K,
        typename M,
        typename H = Hash<K>,
        typename P = std::equal_to<K>,
        typename A = Allocator<pair<const K, M>>>
using HashMap = boost::unordered::unordered_map<K, M, H, P, A>;

template <
        typename K,
        typename M,
        typename H = Hash<K>,
        typename P = std::equal_to<K>,
        typename A = Allocator<pair<const K, M>>>
using FlatHashMap = boost::unordered::unordered_flat_map<K, M, H, P, A>;

template <
        typename K,
        typename M,
        typename H = Hash<K>,
        typename P = std::equal_to<K>,
        typename A = Allocator<pair<const K, M>>>
using FlatNodeMap = boost::unordered::unordered_node_map<K, M, H, P, A>;

}// namespace containers
}// namespace rpy

#endif// ROUHGPY_CORE_CONTAINER_UNORDERD_MAP_H
