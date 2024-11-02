//
// Created by sam on 24/10/24.
//

#include "small_object_memory.h"

#include "AlignedMemory.h"

using namespace rpy;

mem::small::PoolMemory* mem::get_pool_memory() noexcept
{
    static auto memory_resource = std::make_unique<small::PoolMemory>(
            std::pmr::pool_options{
                    small::small_alloc_max_chunks,
                    small::small_alloc_chunk_size
            },
            mem::AlignedMemory::get()
    );
    return memory_resource.get();
}

mem::small::PoolMemory* mem::small::get_small_object_memory_resource() noexcept
{
    return get_pool_memory();
}
