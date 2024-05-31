// Copyright (c) 2023 the R ughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_KERNEL_ARG_H_
#define ROUGHPY_DEVICE_KERNEL_ARG_H_

#include "core.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

/**
 * @class KernelArgument
 * @brief Represents an argument for a kernel function.
 *
 * This class encapsulates various types of kernel arguments, including pointers
 * to data, pointers to constant data, pointers to buffers, or constant pointers
 * to buffers.
 *
 * Usage example:
 * @code
 * Buffer buffer;
 * int data = 42;
 * const float constData = 3.14f;
 *
 * KernelArgument arg1(buffer); // Create an argument from a buffer
 * KernelArgument arg2(data); // Create an argument from non-const data
 * KernelArgument arg3(constData); // Create an argument from constant data
 * @endcode
 */
class ROUGHPY_DEVICES_EXPORT KernelArgument
{
    const void* p_data;

    enum
    {
        Ref,
        CRef,
        Buf,
        CBuf,
    } m_mode;

public:
    KernelArgument(const KernelArgument&) = default;

    KernelArgument(KernelArgument&& other) noexcept
        : p_data(nullptr),
          m_mode(other.m_mode)
    {
        std::swap(p_data, other.p_data);
    }

    explicit KernelArgument(const Buffer& buffer) : p_data(&buffer), m_mode(Buf)
    {}
    KernelArgument(const Reference& value) : p_data(&value), m_mode(Ref) {}

    KernelArgument(const ConstReference& value) : p_data(&value), m_mode(CRef)
    {}

    RPY_NO_DISCARD constexpr bool is_buffer() const noexcept
    {
        return m_mode == Buf || m_mode == CBuf;
    }

    RPY_NO_DISCARD constexpr bool is_ref() const noexcept
    {
        return m_mode == Ref || m_mode == CRef;
    }

    RPY_NO_DISCARD constexpr bool is_const() const noexcept
    {
        return m_mode == CRef || m_mode == CBuf;
    }

    RPY_NO_DISCARD const Reference& ref() const noexcept;

    RPY_NO_DISCARD const ConstReference& cref() const noexcept;

    RPY_NO_DISCARD const Buffer& buffer() const noexcept;

    RPY_NO_DISCARD const Buffer& cbuffer() const noexcept;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_ARG_H_
