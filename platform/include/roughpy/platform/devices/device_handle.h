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

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/filesystem.h>

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <mutex>

#include "core.h"

#include "buffer.h"
#include "core.h"
#include "event.h"
#include "kernel.h"
#include "queue.h"

namespace rpy {
namespace devices {

struct ExtensionSourceAndOptions {
    std::vector<string> sources;
    string compile_options;
    std::vector<pair<string, string>> header_name_and_source;
    string link_options;
};



/**
 * @brief Interface for interacting with compute devices.
 *
 *
 */
class ROUGHPY_PLATFORM_EXPORT DeviceHandle
    : public mem::RcBase<DeviceHandle>
{
    mutable std::recursive_mutex m_lock;
    mutable std::unordered_map<string, Kernel> m_kernel_cache;

protected:
    using lock_type = std::recursive_mutex;
    using guard_type = std::lock_guard<lock_type>;

    lock_type& get_lock() const noexcept { return m_lock; }


public:

    DeviceHandle();

    virtual ~DeviceHandle();

    RPY_NO_DISCARD virtual DeviceType type() const noexcept;
    RPY_NO_DISCARD virtual DeviceCategory category() const noexcept;

    RPY_NO_DISCARD virtual DeviceInfo info() const noexcept;

    RPY_NO_DISCARD virtual optional<fs::path> runtime_library() const noexcept;

    //    virtual void launch_kernel(const void* kernel,
    //                               const void* launch_config,
    //                               void** args
    //                               ) = 0;

    RPY_NO_DISCARD virtual Buffer
    raw_alloc(dimn_t count, dimn_t alignment) const;

    virtual void raw_free(void* pointer, dimn_t size) const;

    virtual bool has_compiler() const noexcept;

    virtual const Kernel& register_kernel(Kernel kernel) const;

    RPY_NO_DISCARD
    virtual optional<Kernel> get_kernel(const string& name) const noexcept;
    RPY_NO_DISCARD
    virtual optional<Kernel>
    compile_kernel_from_str(const ExtensionSourceAndOptions& args) const;

    virtual void compile_kernels_from_src(const ExtensionSourceAndOptions& args
    ) const;

    RPY_NO_DISCARD virtual Event new_event() const;
    RPY_NO_DISCARD virtual Queue new_queue() const;
    RPY_NO_DISCARD virtual Queue get_default_queue() const;

    RPY_NO_DISCARD virtual optional<boost::uuids::uuid> uuid() const noexcept;
    RPY_NO_DISCARD virtual optional<PCIBusInfo> pci_bus_info() const noexcept;

    RPY_NO_DISCARD virtual bool supports_type(const TypeInfo& info
    ) const noexcept;

    RPY_NO_DISCARD virtual Event
    from_host(Buffer& dst, const BufferInterface& src, Queue& queue) const;

    virtual Event to_host(Buffer& dst, const BufferInterface& src, Queue& queue) const;
};



}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_HANDLE_H_
