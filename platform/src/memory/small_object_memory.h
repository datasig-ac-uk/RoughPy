//
// Created by sam on 24/10/24.
//

#ifndef SMALL_OBJECT_MEMORY_H
#define SMALL_OBJECT_MEMORY_H


#include "memory.h"

namespace rpy {

using PoolMemory = std::pmr::synchronized_pool_resource;

PoolMemory* get_pool_memory() noexcept;


}

#endif //SMALL_OBJECT_MEMORY_H
