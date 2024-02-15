// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_MEMORY_VIEW_H_
#define ROUGHPY_DEVICE_MEMORY_VIEW_H_

#include "core.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

/**
 * @brief A read-only view into mapped data that might live on a separate device
 *
 * This object is a read-only view into a region of memory mapped into main RAM.
 * The data itself might live on device or in system RAM, or it could be backed
 * by a memory-mapped file of some description. MemoryViews retain a strong
 * reference to their mapped buffer, to make sure the data is valid even if the
 * original object is destroyed. It should not be possible to modify the
 * buffer-held data whilst a MemoryView is alive, but if this does happen then
 * then the changes might not be reflected in the view itself.
 *
 */
class ROUGHPY_PLATFORM_EXPORT MemoryView
{
    Buffer m_memory_owner;
    const void* p_data = nullptr;
    dimn_t m_size = 0;

    friend class Buffer;

    /*
     * We don't want people to construct their own view objects.
     * To that end, we make the constructor private, and befriend the Buffer
     * type so it can create buffers.
     */
    MemoryView(const Buffer& buf, const void* data, dimn_t size)
        : m_memory_owner(buf),
          p_data(data),
          m_size(size)
    {}

public:

    ~MemoryView();

    constexpr const void* raw_ptr(dimn_t offset = 0) const noexcept
    {
        return static_cast<const byte*>(p_data) + offset;
    }
    constexpr dimn_t size() const noexcept { return m_size; }

    MemoryView slice(dimn_t offset_bytes, dimn_t size_bytes)
    {
        RPY_DBG_ASSERT(offset_bytes + size_bytes <= m_size);
        return {m_memory_owner, raw_ptr(offset_bytes), size_bytes};
    }

    template <typename T>
    Slice<const T> as_slice() const noexcept
    {
        RPY_DBG_ASSERT(m_size % sizeof(T) == 0);
        return {static_cast<const T*>(p_data), m_size / sizeof(T)};
    }


    bool operator==(const MemoryView& other) const noexcept
    {
        return (m_memory_owner == other.m_memory_owner);
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_MEMORY_VIEW_H_
