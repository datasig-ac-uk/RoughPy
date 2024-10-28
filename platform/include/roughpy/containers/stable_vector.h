//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_STABLE_VECTOR_H
#define ROUGHPY_PLATFORM_CONTAINERS_STABLE_VECTOR_H


#include <boost/container/stable_vector.hpp>

#include "roughpy/platform/memory.h"

namespace rpy {

template <typename T, typename Alloc = mem::AlignedAllocator<T>>
using StableVector = boost::container::stable_vector<T, Alloc>;

}


#endif //ROUGHPY_PLATFORM_CONTAINERS_STABLE_VECTOR_H
