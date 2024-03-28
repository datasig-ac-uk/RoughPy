//
// Created by sam on 3/28/24.
//

#ifndef ROUGHPY_CORE_CONTAINER_VECTOR_H
#define ROUGHPY_CORE_CONTAINER_VECTOR_H

#include "roughpy/core/alloc.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <boost/container/small_vector.hpp>

#include <vector>

namespace rpy {
namespace containers {

template <typename T, typename Alloc = Allocator<T>>
using Vec = std::vector<T, Alloc>;

template <typename T, size_t N, typename Alloc = Allocator<T>>
using SmallVec = boost::container::small_vector<T, N, Alloc>;

}// namespace containers
}// namespace rpy

#endif// ROUGHPY_CORE_CONTAINER_VECTOR_H
