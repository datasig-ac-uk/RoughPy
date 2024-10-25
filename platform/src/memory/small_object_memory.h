//
// Created by sam on 24/10/24.
//

#ifndef SMALL_OBJECT_MEMORY_H
#define SMALL_OBJECT_MEMORY_H


#include "memory.h"

namespace rpy { namespace mem {



small::PoolMemory* get_pool_memory() noexcept;

} //namespace mem
}

#endif //SMALL_OBJECT_MEMORY_H
