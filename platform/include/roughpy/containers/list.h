//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_LIST_H
#define ROUGHPY_PLATFORM_CONTAINERS_LIST_H


#include <list>

#include "roughpy/platform/memory.h"

namespace rpy {

template <typename T, typename Allocator = mem::small::PoolAllocator<T>>
using List = std::list<T, Allocator>;

}


#endif //ROUGHPY_PLATFORM_CONTAINERS_LIST_H
