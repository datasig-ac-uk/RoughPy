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

#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/check.h>
#include <roughpy/core/slice.h>
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy {
namespace devices {

class ROUGHPY_PLATFORM_EXPORT MemoryView
{
    Buffer& r_memory_owner;
    void* p_data = nullptr;
    dimn_t m_size = 0;
    BufferMode m_mode;

    friend class Buffer;

    /*
     * We don't want people to construct their own view objects.
     * To that end, we make the constructor private, and befriend the Buffer
     * type so it can create buffers.
     */
    MemoryView(Buffer& buf, void* data, dimn_t size, BufferMode mode)
        : r_memory_owner(buf),
          p_data(data),
          m_size(size),
          m_mode(mode)
    {}

public:

    ~MemoryView();

    constexpr void* raw_ptr() noexcept { return p_data; }
    constexpr dimn_t size() const noexcept { return m_size; }
    constexpr BufferMode mode() const noexcept { return m_mode; }

    template <typename T>
    Slice<const T> as_slice() const noexcept
    {
        RPY_DBG_ASSERT(m_size % sizeof(T) == 0);
        return {static_cast<const T*>(p_data), m_size / sizeof(T)};
    }

    template <typename T>
    Slice<T> as_mut_slice()
    {
        RPY_DBG_ASSERT(m_size % sizeof(T) == 0);
        RPY_CHECK(m_mode != BufferMode::Read);
        return {static_cast<T*>(p_data), m_size / sizeof(T)};
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_MEMORY_VIEW_H_
