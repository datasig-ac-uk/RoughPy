//
// Created by sam on 25/10/24.
//

#ifndef ROUGHPY_PLATFORM_CONTAINERS_VECTOR_H
#define ROUGHPY_PLATFORM_CONTAINERS_VECTOR_H


#include <vector>

#include "roughpy/platform/memory.h"

namespace rpy {

template <typename T, typename Alloc = std::allocator<T>>
using Vec = std::vector<T, Alloc>;

}


#endif //ROUGHPY_PLATFORM_CONTAINERS_VECTOR_H
