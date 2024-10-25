//
// Created by sam on 24/10/24.
//

#include "small_object_memory.h"

#include "AlignedMemory.h"

using namespace rpy;


rpy::PoolMemory* rpy::get_pool_memory() noexcept
{
    static PoolMemory memory_resource(
            std::pmr::pool_options{mem::small_alloc_chunk_size},
            mem::AlignedMemory::get()
    );
    return &memory_resource;
}

std::pmr::memory_resource* mem::get_small_object_memory_resource() noexcept
{
    return rpy::get_pool_memory();
}
