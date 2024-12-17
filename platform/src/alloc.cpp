//
// Created by sammorley on 19/11/24.
//

#include "roughpy/platform/alloc.h"

#include <new>

#include <boost/pool/singleton_pool.hpp>
#include <boost/align/aligned_alloc.hpp>
#include <roughpy/core/traits.h>

using namespace rpy;
using namespace rpy::mem;


void* rpy::mem::aligned_alloc(size_t alignment, size_t size) noexcept
{
    return boost::alignment::aligned_alloc(alignment, size);
}

void rpy::mem::aligned_free(void* ptr, size_t size) noexcept
{
    ignore_unused(size);
    boost::alignment::aligned_free(ptr);
}

//
// namespace {
//
// class PageAlignedMemoryResource : public std::pmr::memory_resource
// {
//
// protected:
//     void* do_allocate(size_t bytes, size_t alignment) override
//     {
//         ignore_unused(alignment);
//         void* ptr = mem::aligned_alloc(small_chunk_size, bytes);
//         if (!ptr) { throw std::bad_alloc(); }
//         return ptr;
//     }
//
//     void do_deallocate(void* p, size_t bytes, size_t alignment) override
//     {
//         ignore_unused(alignment);
//         return mem::aligned_free(p, bytes);
//     }
//
//     bool do_is_equal(const std::pmr::memory_resource& other
//     ) const noexcept override
//     {
//         return this == &other;
//     }
//
// public:
//     static PageAlignedMemoryResource* get() noexcept
//     {
//         static PageAlignedMemoryResource instance;
//         return &instance;
//     }
// };
//
// std::pmr::synchronized_pool_resource* get_pool() noexcept
// {
//     static std::pmr::synchronized_pool_resource pool (
//         std::pmr::pool_options{ small_blocks_per_chunk, small_block_size },
//         PageAlignedMemoryResource::get());
//     return &pool;
// }
// }


namespace {


struct MyAlignedAlloc
{
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    static char* malloc BOOST_PREVENT_MACRO_SUBSTITUTION(const size_type size) noexcept
    {
        return static_cast<char*>(mem::aligned_alloc(small_chunk_size, size));
    }

    static void free BOOST_PREVENT_MACRO_SUBSTITUTION(char* const block) noexcept
    {
        mem::aligned_free(block, small_chunk_size);
    }

};

boost::pool<MyAlignedAlloc>* get_pool() noexcept
{
    static boost::pool<MyAlignedAlloc> pool {small_block_size, small_blocks_per_chunk};
    return &pool;
}


}


void* rpy::mem::small_object_alloc(size_t size)
{
    static constexpr size_t max_alignment = alignof(std::max_align_t);
    // return get_pool()->allocate(size);
    if (size > small_block_size) {
        return aligned_alloc(max_alignment, size);
    }
    return get_pool()->malloc();
}

void rpy::mem::small_object_free(void* ptr, size_t size)
{
    if (size > small_block_size) {
        aligned_free(ptr, size);
    } else {
        get_pool()->free(static_cast<char*>(ptr));
    }

    // get_pool()->deallocate(ptr, size);
}

void* SmallObjectBase::operator new(size_t size)
{
    return small_object_alloc(size);
}

void SmallObjectBase::operator delete(void* object, size_t size)
{
    small_object_free(object, size);
}