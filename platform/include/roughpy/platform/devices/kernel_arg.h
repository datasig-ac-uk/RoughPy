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

#ifndef ROUGHPY_DEVICE_KERNEL_ARG_H_
#define ROUGHPY_DEVICE_KERNEL_ARG_H_


#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "roughpy/platform/roughpy_platform_export.h"

#include "core.h"

namespace rpy {
namespace devices {

class ROUGHPY_PLATFORM_EXPORT KernelArgument
{

    union
    {
        void* p_data;
        const void* p_const_data;
        Buffer* p_buffer;
        const Buffer* p_const_buffer;
    };

    enum
    {
        Pointer,
        ConstPointer,
        BufferPointer,
        ConstBufferPointer
    } m_mode;

    TypeInfo m_info;

public:
    explicit KernelArgument(Buffer& buffer)
        : p_buffer(&buffer),
          m_mode(BufferPointer)
    {}

    explicit KernelArgument(const Buffer& buffer)
        : p_const_buffer(&buffer),
          m_mode(BufferPointer)
    {}

    template <typename T>
    explicit KernelArgument(T& data)
        : p_data(&data),
          m_mode(Pointer),
          m_info(type_info<T>())
    {}

    template <typename T>
    explicit KernelArgument(const T& data)
        : p_const_data(&data),
          m_mode(ConstPointer),
          m_info(type_info<T>())
    {}

    RPY_NO_DISCARD constexpr bool is_buffer() const noexcept
    {
        return m_mode == BufferPointer || m_mode == ConstBufferPointer;
    }

    RPY_NO_DISCARD constexpr bool is_const() const noexcept
    {
        return m_mode == ConstPointer || m_mode == ConstBufferPointer;
    }

    RPY_NO_DISCARD constexpr void* pointer() const noexcept
    {
        RPY_DBG_ASSERT(m_mode == Pointer);
        return p_data;
    }

    RPY_NO_DISCARD constexpr const void* const_pointer() const noexcept
    {
        RPY_DBG_ASSERT(!is_buffer());
        return (m_mode == Pointer) ? p_data : p_const_data;
    }

    RPY_NO_DISCARD constexpr Buffer& buffer() const noexcept
    {
        RPY_DBG_ASSERT(m_mode == BufferPointer);
        return *p_buffer;
    }

    RPY_NO_DISCARD constexpr const Buffer& const_buffer() const noexcept
    {
        RPY_DBG_ASSERT(is_buffer());
        return (m_mode == BufferPointer) ? static_cast<const Buffer&>(*p_buffer)
                                         : *p_const_buffer;
    }

    RPY_NO_DISCARD constexpr dimn_t size() const noexcept
    {
        return (is_buffer()) ? sizeof(void*) : m_info.bytes;
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_ARG_H_
