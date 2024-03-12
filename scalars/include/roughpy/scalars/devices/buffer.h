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
    /*
     * I'd really like to have the memory owner pointer here, but I can't
     * because the Buffer class isn't defined until later. Instead, the
     * implementations will have to handle this by themselves.
     */
public:
    using object_t = Buffer;

    RPY_NO_DISCARD virtual BufferMode mode() const;
    RPY_NO_DISCARD virtual TypeInfo type_info() const noexcept;
    RPY_NO_DISCARD virtual dimn_t size() const;
    RPY_NO_DISCARD virtual dimn_t bytes() const;

    RPY_NO_DISCARD virtual Event
    to_device(Buffer& dst, const Device& device, Queue& queue);

    RPY_NO_DISCARD virtual Buffer map_mut(dimn_t size, dimn_t offset);
    RPY_NO_DISCARD virtual Buffer map(dimn_t size, dimn_t offset) const;

    virtual void unmap(BufferInterface& ptr) const noexcept;

    virtual Buffer memory_owner() const noexcept;

    virtual Buffer slice(dimn_t offset, dimn_t size) const;
    virtual Buffer mut_slice(dimn_t offset, dimn_t size);
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

    Buffer(Device device, dimn_t size, TypeInfo type);
    Buffer(dimn_t size, TypeInfo type);
    Buffer(Device device, void* ptr, dimn_t size, TypeInfo type);
    Buffer(Device device, const void* ptr, dimn_t size, TypeInfo type);

    Buffer(void* ptr, dimn_t size, TypeInfo info);
    Buffer(const void* ptr, dimn_t size, TypeInfo info);

    template <typename T>
    explicit Buffer(Device device, Slice<T> data);

    template <typename T>
    explicit Buffer(Device device, Slice<const T> data);

    template <typename T>
    explicit Buffer(Slice<T> data);

    template <typename T>
    explicit Buffer(Slice<const T> data);

    RPY_NO_DISCARD dimn_t size() const;
    RPY_NO_DISCARD dimn_t bytes() const;
    RPY_NO_DISCARD TypeInfo type_info() const noexcept;
    RPY_NO_DISCARD BufferMode mode() const;

    template <typename T>
    Slice<const T> as_slice() const
    {
        return {static_cast<const T*>(ptr()), size()};
    }

    template <typename T>
    Slice<T> as_mut_slice()
    {
        RPY_CHECK(mode() != BufferMode::Read);
        return {static_cast<T*>(ptr()), size()};
    }

    RPY_NO_DISCARD Buffer slice(dimn_t offset, dimn_t size);
    RPY_NO_DISCARD Buffer slice(dimn_t offset, dimn_t size) const;

    void to_device(Buffer& dst, const Device& device);

    Event to_device(Buffer& dst, const Device& device, Queue& queue);

    RPY_NO_DISCARD Buffer map(dimn_t size = 0, dimn_t offset = 0) const;
    RPY_NO_DISCARD Buffer map(dimn_t size = 0, dimn_t offset = 0);

    Buffer memory_owner() const noexcept;

    bool is_owner() const noexcept
    {
        // This is a really bad implementation, but it will do for now
        return memory_owner().impl() == impl();
    }
};

template <typename T>
Buffer::Buffer(rpy::devices::Device device, Slice<T> data)
    : Buffer(device, data.data(), data.size(), devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(rpy::devices::Device device, Slice<const T> data)
    : Buffer(device, data.data(), data.size(), devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(Slice<T> data)
    : Buffer(get_host_device(),
             data.data(),
             data.size(),
             devices::type_info<T>())
{}

template <typename T>
Buffer::Buffer(Slice<const T> data)
    : Buffer(get_host_device(),
             data.data(),
             data.size(),
             devices::type_info<T>())
{}

}// namespace devices
}// namespace rpy
#endif// ROUGHPY_DEVICE_BUFFER_H_
