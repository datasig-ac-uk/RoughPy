//
// Created by sam on 24/10/24.
//

#include "AlignedMemory.h"

using namespace rpy;

void* AlignedMemory::do_allocate(std::size_t bytes, std::size_t alignment)
{
    return dtl::aligned_alloc(alignment, bytes);
}

void AlignedMemory::do_deallocate(
        void* p,
        std::size_t bytes,
        std::size_t alignment
)
{
    return dtl::aligned_free(p, bytes);
}

bool AlignedMemory::do_is_equal(const memory_resource& other) const noexcept
{
    return true;
}

AlignedMemory* AlignedMemory::get() noexcept
{
    static AlignedMemory s_aligned_memory;
    return &s_aligned_memory;
}


std::pmr::memory_resource* rpy::get_base_memory_resource() noexcept
{
    return AlignedMemory::get();
}
