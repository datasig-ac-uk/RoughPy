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

#include "core.h"

#include <roughpy/core/traits.h>
#include <roughpy/platform/filesystem.h>

namespace rpy { namespace device {

class RPY_EXPORT DeviceHandle
{
    DeviceInfo m_info;

public:
    virtual ~DeviceHandle();

    explicit DeviceHandle(DeviceInfo info) : m_info(std::move(info)) {}

    explicit DeviceHandle(DeviceType type, int32_t device_id)
            : m_info {type, device_id}
    {}

    RPY_NO_DISCARD const DeviceInfo& info() const noexcept { return m_info; }
    //
    RPY_NO_DISCARD
    virtual optional<fs::path> runtime_library() const noexcept;

    //    virtual void launch_kernel(const void* kernel,
    //                               const void* launch_config,
    //                               void** args
    //                               ) = 0;

    RPY_NO_DISCARD virtual void*
    raw_allocate(dimn_t size, dimn_t alignment) const = 0;

    virtual void raw_dealloc(void* d_raw_pointer, dimn_t size) const = 0;

    virtual void
    copy_to_device(void* d_dst_raw, const void* h_src_raw, dimn_t count)
            const = 0;

    virtual void copy_from_device(
            void* h_dst_raw, const void* d_src_raw, dimn_t count
    ) const = 0;


    virtual Kernel* get_kernel(string_view name) const = 0;

};

constexpr bool operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}



}}



#endif // ROUGHPY_DEVICE_DEVICE_HANDLE_H_
