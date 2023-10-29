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

#ifndef ROUGHPY_DEVICE_BUFFER_H_
#define ROUGHPY_DEVICE_BUFFER_H_

#include "core.h"
#include "device_object_base.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

enum class BufferMode
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3
};

class BufferInterface : public dtl::InterfaceBase
{

public:
    using object_t = Buffer;


    RPY_NO_DISCARD virtual BufferMode mode() const;

    RPY_NO_DISCARD virtual dimn_t size() const;

    RPY_NO_DISCARD virtual Event
    to_device(Buffer& dst, const Device& device, Queue& queue);
};

class Buffer : public dtl::ObjectBase<BufferInterface, Buffer>
{
    using base_t = dtl::ObjectBase<BufferInterface, Buffer>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD dimn_t size() const;

    RPY_NO_DISCARD BufferMode mode() const;

    template <typename T>
    Slice<const T> as_slice() const
    {
        return {static_cast<const T*>(ptr()), size() / sizeof(T)};
    }

    void to_device(Buffer& dst, const Device& device);

    Event to_device(Buffer& dst, const Device& device, Queue& queue);
};


}// namespace devices
}// namespace rpy
#endif// ROUGHPY_DEVICE_BUFFER_H_
