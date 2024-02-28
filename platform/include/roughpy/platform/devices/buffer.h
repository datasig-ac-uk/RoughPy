// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_DEVICE_BUFFER_H_
#define ROUGHPY_DEVICE_BUFFER_H_

#include "core.h"
#include "device_object_base.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

class ROUGHPY_PLATFORM_EXPORT BufferInterface : public dtl::InterfaceBase
{

public:
    using object_t = Buffer;

    RPY_NO_DISCARD virtual BufferMode mode() const;

    RPY_NO_DISCARD virtual dimn_t size() const;

    RPY_NO_DISCARD virtual Event
    to_device(Buffer& dst, const Device& device, Queue& queue);

    RPY_NO_DISCARD virtual void*
    map(BufferMode map_mode, dimn_t size, dimn_t offset);

    virtual void unmap(void* ptr) noexcept;
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<BufferInterface, Buffer>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<BufferInterface, Buffer>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_PLATFORM_EXPORT
        ObjectBase<BufferInterface, Buffer>;
}
#endif

class ROUGHPY_PLATFORM_EXPORT Buffer
    : public dtl::ObjectBase<BufferInterface, Buffer>
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

    template <typename T>
    Slice<T> as_mut_slice()
    {
        RPY_CHECK(mode() != BufferMode::Read);
        return {static_cast<T*>(ptr()), size() / sizeof(T)};
    }

    void to_device(Buffer& dst, const Device& device);

    Event to_device(Buffer& dst, const Device& device, Queue& queue);

    RPY_NO_DISCARD MemoryView
    map(BufferMode map_mode = BufferMode::None,
        dimn_t size = 0,
        dimn_t offset = 0);
    void unmap(MemoryView& view) noexcept;
};

}// namespace devices
}// namespace rpy
#endif// ROUGHPY_DEVICE_BUFFER_H_
