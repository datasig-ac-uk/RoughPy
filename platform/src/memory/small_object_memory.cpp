//
// Created by sam on 24/10/24.
//

#include "small_object_memory.h"

#include "AlignedMemory.h"

rpy::PoolMemory* rpy::get_pool_memory() noexcept
{
    static PoolMemory memory_resource(
            std::pmr::pool_options{rpy::small_alloc_chunk_size},
            AlignedMemory::get()
    );
    return &memory_resource;
}

std::pmr::memory_resource* rpy::get_small_object_memory_resource() noexcept
{
    return rpy::get_pool_memory();
}
