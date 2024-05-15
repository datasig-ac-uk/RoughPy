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

//
// Created by user on 16/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_DEVICE_H_
#define ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_DEVICE_H_

#include "devices/core.h"
#include "devices/host_device.h"

#include "host_decls.h"

#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

class CPUDeviceHandle : public HostDeviceHandle
{

    CPUDeviceHandle();
    ~CPUDeviceHandle() override;

public:
    static CPUDevice get();

    bool is_host() const noexcept override;
    DeviceCategory category() const noexcept override;
    DeviceInfo info() const noexcept override;

    Buffer alloc(TypeInfo info, dimn_t count) const override;
    RPY_NO_DISCARD Buffer alloc(const Type* type, dimn_t count) const override;

    void raw_free(Buffer& buf) const override;
    RawBuffer allocate_raw_buffer(dimn_t size, dimn_t alignment) const;
    void free_raw_buffer(RawBuffer& buffer) const;

    bool has_compiler() const noexcept override;
    optional<Kernel> get_kernel(const string& name) const noexcept override;
    optional<Kernel>
    compile_kernel_from_str(const ExtensionSourceAndOptions& args
    ) const override;
    void compile_kernels_from_src(const ExtensionSourceAndOptions& args
    ) const override;
    Event new_event() const override;
    Queue new_queue() const override;
    Queue get_default_queue() const override;
    bool supports_type(const TypeInfo& info) const noexcept override;

    Device compute_delegate() const override;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_DEVICE_H_
