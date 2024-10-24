//
// Created by sam on 24/10/24.
//

#ifndef ALIGNEDMEMORY_H
#define ALIGNEDMEMORY_H


#include "memory.h"


namespace rpy {

class AlignedMemory : public std::pmr::memory_resource {
protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment)
            override;

    RPY_NO_DISCARD
    bool do_is_equal(const memory_resource& other) const noexcept override;

public:
    static AlignedMemory* get() noexcept;
};

} // rpy

#endif //ALIGNEDMEMORY_H
