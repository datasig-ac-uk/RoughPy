//
// Created by sam on 24/10/24.
//

#include "small_object_memory.h"

#include "AlignedMemory.h"

using namespace rpy;


mem::small::PoolMemory* mem::get_pool_memory() noexcept
{
    static mem::small::PoolMemory memory_resource(
            std::pmr::pool_options{mem::small_alloc_chunk_size},
            mem::AlignedMemory::get()
    );
    return &memory_resource;
}

mem::small::PoolMemory* mem::get_small_object_memory_resource() noexcept
{
    return get_pool_memory();
}
