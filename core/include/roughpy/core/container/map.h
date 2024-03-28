//
// Created by sam on 3/26/24.
//

#ifndef ROUHGPY_CORE_CONTAINER_MAP_H
#define ROUHGPY_CORE_CONTAINER_MAP_H

#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <boost/container/flat_map.hpp>
#include <boost/container/map.hpp>
#include <boost/container/options.hpp>
#include <boost/container/small_vector.hpp>

#include <functional>

namespace rpy {
namespace containers {


template <
        typename Key,
        typename Mapped,
        typename Compare = std::less<Key>,
        typename Alloc = Allocator<pair<const Key, Mapped>>,
        typename Options = boost::container::tree_assoc_defaults>
using Map = boost::container::map<Key, Mapped, Compare, Alloc, Options>;



template <
        typename Key,
        typename Mapped,
        typename Compare = std::less<Key>,
        typename Alloc = Allocator<pair<const Key, Mapped>>>
using FlatMap = boost::container::flat_map<Key, Mapped, Compare, Alloc>;



template <
        typename Key,
        typename Mapped,
        dimn_t SmallSize,
        typename Compare = std::less<Key>,
        typename Alloc = Allocator<pair<const Key, Mapped>>>
using SmallFlatMap = boost::container::flat_map<
        Key,
        Mapped,
        Compare,
        boost::container::
                small_vector<pair<const Key, Mapped>, SmallSize, Alloc>>;

}// namespace containers
}// namespace rpy

#endif// ROUHGPY_CORE_CONTAINER_MAP_H
