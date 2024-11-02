//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_DEQUE_H
#define ROUGHPY_PLATFORM_CONTAINERS_DEQUE_H


#include <deque>

#include "roughpy/platform/memory.h"

namespace rpy {

template <typename T, typename Alloc=std::allocator<T>>
using Deque = std::deque<T, Alloc>;

}


#endif //ROUGHPY_PLATFORM_CONTAINERS_DEQUE_H
