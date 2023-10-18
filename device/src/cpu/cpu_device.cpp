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
// Created by user on 16/10/23.
//

#include "cpu_device.h"

#include <roughpy/core/alloc.h>

#include <roughpy/device/buffer.h>
#include <roughpy/device/event.h>
#include <roughpy/device/kernel.h>
#include <roughpy/device/queue.h>


#include <boost/container/small_vector.hpp>

#include "opencl/ocl_buffer.h"
#include "opencl/ocl_device.h"
#include "opencl/ocl_event.h"
#include "opencl/ocl_handle_errors.h"
#include "opencl/ocl_kernel.h"
#include "opencl/ocl_queue.h"
#include "opencl/ocl_helpers.h"
#include "cpu_buffer.h"
#include "cpu_event.h"
#include "cpu_kernel.h"
#include "cpu_queue.h"

using namespace rpy;
using namespace rpy::devices;

namespace bc = boost::container;

CPUDeviceHandle::CPUDeviceHandle() : p_ocl_handle(nullptr)
{

    cl_uint num_platforms = 0;
    auto ecode = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    if (num_platforms > 0) {
        bc::small_vector<cl_platform_id, 1> platforms(num_platforms);

        ecode = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

        bc::small_vector<cl_device_id, 1> candidates;

        auto clear_candidates = [&candidates]() {
            for (auto&& candidate : candidates) {
                clReleaseDevice(candidate);
            }
            candidates.clear();
        };

        cl_uint num_devices = 0;
        for (auto&& platform : platforms) {
            ecode = clGetDeviceIDs(
                    platform,
                    CL_DEVICE_TYPE_CPU,
                    0,
                    nullptr,
                    &num_devices
            );
            if (ecode != CL_SUCCESS) {
                clear_candidates();
                RPY_HANDLE_OCL_ERROR(ecode);
            }

            if (num_devices > 0) {
                auto current_size = candidates.size();
                candidates.resize(current_size + num_devices);
                ecode = clGetDeviceIDs(
                        platform,
                        CL_DEVICE_TYPE_CPU,
                        num_devices,
                        candidates.data() + current_size,
                        nullptr
                );
                if (ecode != CL_SUCCESS) {
                    clear_candidates();
                    RPY_HANDLE_OCL_ERROR(ecode);
                }
            }
        }

        if (!candidates.empty()) {
            // TODO: more sophisticated logic for choosing the best
            //  implementation of OpenCL to use. For now, just pick the first
            //  one.
            p_ocl_handle = new OCLDeviceHandle(candidates[0]);

            for (size_t i=1; i<candidates.size(); ++i) {
                clReleaseDevice(candidates[i]);
            }
            candidates.clear();
        }

    }
}
CPUDeviceHandle::~CPUDeviceHandle() = default;

CPUDevice CPUDeviceHandle::get() {
    static const CPUDeviceHandle device;
    return &device;
}



DeviceInfo CPUDeviceHandle::info() const noexcept
{
    return {DeviceType::CPU, 0};
}

Buffer CPUDeviceHandle::raw_alloc(dimn_t count, dimn_t alignment) const
{
    return DeviceHandle::raw_alloc(count, alignment);
}
void CPUDeviceHandle::raw_free(Buffer buffer) const {}
optional<Kernel> CPUDeviceHandle::get_kernel(string_view name) const noexcept
{
    return DeviceHandle::get_kernel(name);
}
optional<Kernel> CPUDeviceHandle::compile_kernel_from_str(string_view code
) const
{
    return DeviceHandle::compile_kernel_from_str(code);
}
void CPUDeviceHandle::compile_kernels_from_src(string_view code) const
{
    DeviceHandle::compile_kernels_from_src(code);
}
Event CPUDeviceHandle::new_event() const { return DeviceHandle::new_event(); }
Queue CPUDeviceHandle::new_queue() const { return DeviceHandle::new_queue(); }
Queue CPUDeviceHandle::get_default_queue() const
{
    return Queue();
}
bool CPUDeviceHandle::supports_type(const TypeInfo& info) const noexcept
{
    return true;
}
OCLDevice CPUDeviceHandle::ocl_device() const noexcept
{
    return p_ocl_handle;
}
