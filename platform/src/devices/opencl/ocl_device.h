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

//
// Created by user on 11/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_DEVICE_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_DEVICE_H_

#include "ocl_buffer.h"
#include "ocl_decls.h"
#include "ocl_event.h"
#include "ocl_headers.h"
#include "ocl_kernel.h"
#include "ocl_queue.h"
#include "ocl_version.h"

#include "devices/device_handle.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace rpy {
namespace devices {

class OCLDeviceHandle : public DeviceHandle
{
    cl_device_id m_device;
    int32_t m_device_id;

    cl_context m_ctx;
    cl_command_queue m_default_queue;

    OCLVersion m_ocl_version;

    mutable std::vector<cl_program> m_programs;
    mutable std::unordered_map<string, cl_program> m_header_cache;



    using typename DeviceHandle::guard_type;


public:
    explicit OCLDeviceHandle(cl_device_id id);

    ~OCLDeviceHandle() override;

    DeviceType type() const noexcept override;
    DeviceCategory category() const noexcept override;
    DeviceInfo info() const noexcept override;
    optional<fs::path> runtime_library() const noexcept override;
    Buffer raw_alloc(dimn_t count, dimn_t alignment) const override;
    void raw_free(void* pointer, dimn_t size) const override;

    bool has_compiler() const noexcept override;

private:
    cl_program
    get_header_program(const string& name, const string& source) const;
    cl_program compile_program(const ExtensionSourceAndOptions& args) const;

public:
    optional<Kernel>
    compile_kernel_from_str(const ExtensionSourceAndOptions& args
    ) const override;
    optional<Kernel> get_kernel(const string& name) const noexcept override;
    void compile_kernels_from_src(const ExtensionSourceAndOptions& args
    ) const override;

    cl_command_queue default_queue() const noexcept { return m_default_queue; }

    cl_context context() const noexcept { return m_ctx; }

    Event new_event() const override;
    Queue new_queue() const override;
    Queue get_default_queue() const override;

    optional<boost::uuids::uuid> uuid() const noexcept override;
    optional<PCIBusInfo> pci_bus_info() const noexcept override;




    Event from_host(Buffer& dst, const BufferInterface& src, Queue& queue)
            const override;
    Event to_host(Buffer& dst, const BufferInterface& src, Queue& queue) const override;

    RPY_NO_DISCARD
    bool is_cpu() const;

    RPY_NO_DISCARD
    cl_platform_id get_platform() const;
    RPY_NO_DISCARD
    bool cl_supports_version(OCLVersion version) const;

    RPY_NO_DISCARD
    Buffer make_buffer(cl_mem buffer, bool move = false) const;
    RPY_NO_DISCARD
    Event make_event(cl_event event, bool move = true) const;
    RPY_NO_DISCARD
    Kernel make_kernel(cl_kernel kernel, bool move = false) const;
    RPY_NO_DISCARD
    Queue make_queue(cl_command_queue queue, bool move = true) const;

};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_DEVICE_H_
