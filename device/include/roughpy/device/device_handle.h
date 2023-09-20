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

#ifndef ROUGHPY_DEVICE_DEVICE_HANDLE_H_
#define ROUGHPY_DEVICE_DEVICE_HANDLE_H_



#include <roughpy/core/traits.h>
#include <roughpy/platform/filesystem.h>

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <memory>

#include "core.h"
#include "buffer.h"
#include "event.h"
#include "kernel.h"
#include "queue.h"


namespace rpy {
namespace device {

class RPY_EXPORT DeviceHandle : public
                                boost::intrusive_ref_counter<DeviceHandle>
{
    const BufferInterface* p_buffer_interface;
    const EventInterface* p_event_interface;
    const KernelInterface* p_kernel_interface;
    const QueueInterface* p_queue_interface;

protected:
    RPY_NO_DISCARD virtual const BufferInterface* buffer_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_buffer_interface);
        return p_buffer_interface;
    }
    RPY_NO_DISCARD virtual const EventInterface* event_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_event_interface);
        return p_event_interface;
    }
    RPY_NO_DISCARD virtual const KernelInterface* kernel_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_kernel_interface);
        return p_kernel_interface;
    }
    RPY_NO_DISCARD virtual const QueueInterface* queue_interface() const noexcept
    {
        RPY_DBG_ASSERT(p_queue_interface);
        return p_queue_interface;
    }

public:
    DeviceHandle();

    DeviceHandle(
            const BufferInterface* buffer_iface,
            const EventInterface* event_iface,
            const KernelInterface* kernel_iface,
            const QueueInterface* queue_iface
    )
        : p_buffer_interface(buffer_iface),
          p_event_interface(event_iface),
          p_kernel_interface(kernel_iface),
          p_queue_interface(queue_iface)
    {}


    virtual ~DeviceHandle();

    RPY_NO_DISCARD virtual DeviceInfo info() const noexcept;

    RPY_NO_DISCARD virtual optional<fs::path> runtime_library() const noexcept;

    //    virtual void launch_kernel(const void* kernel,
    //                               const void* launch_config,
    //                               void** args
    //                               ) = 0;


    RPY_NO_DISCARD virtual Buffer raw_alloc(dimn_t count, dimn_t alignment) const;

    virtual void raw_free(Buffer buffer) const;

};

constexpr bool operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}



}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_HANDLE_H_
