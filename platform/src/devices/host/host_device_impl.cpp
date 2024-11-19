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

#include "host_device_impl.h"

#include <roughpy/core/check.h>
#include <roughpy/core/smart_ptr.h>

#include "roughpy/platform/alloc.h"

#include "devices/buffer.h"
#include "devices/event.h"
#include "devices/kernel.h"
#include "devices/queue.h"

#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>

#include "host_buffer.h"
#include "host_event.h"
#include "host_kernel.h"
#include "host_queue.h"
#include "devices/opencl/ocl_buffer.h"
#include "devices/opencl/ocl_device.h"
#include "devices/opencl/ocl_event.h"
#include "devices/opencl/ocl_handle_errors.h"
#include "devices/opencl/ocl_helpers.h"
#include "devices/opencl/ocl_kernel.h"
#include "devices/opencl/ocl_queue.h"

#include "kernels/masked_binary.h"
#include "kernels/masked_unary.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::devices;

namespace bc = boost::container;

std::atomic_size_t* CPUDeviceHandle::get_ref_count() const
{
    const guard_type access(get_lock());

    if (!m_ref_counts.empty()) {
        auto found = std::find_if(
                m_ref_counts.begin(),
                m_ref_counts.end(),
                [](const std::atomic_size_t& rc) {
                    // We already hold a mutex, so only one thread is doing this
                    return rc.load(std::memory_order_relaxed) == 0;
                }
        );

        if (found != m_ref_counts.end()) {
            // increment now, so there is no risk of a double use.
            found->fetch_add(1, std::memory_order_relaxed);
            return &*found;
        }
    }

    m_ref_counts.emplace_back(1);
    return &m_ref_counts.back();
}

CPUDeviceHandle::CPUDeviceHandle() : p_ocl_handle(nullptr)
{
    const guard_type access(get_lock());

    cl_uint num_platforms = 0;
    auto ecode = clGetPlatformIDs(0, nullptr, &num_platforms);

    if (ecode != CL_SUCCESS) { return; }

    if (num_platforms == 0) { return; }

    bc::small_vector<cl_platform_id, 1> platforms(num_platforms);

    ecode = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (ecode != CL_SUCCESS) { return; }

    bc::small_vector<cl_device_id, 1> candidates;

    auto clear_candidates = [&candidates]() {
        for (auto&& candidate : candidates) { clReleaseDevice(candidate); }
        candidates.clear();
    };

    bc::small_vector<cl_device_id, 1> devices;

    auto clear_devices = [&devices]() {
        for (auto&& dev : devices) { clReleaseDevice(dev); }
        devices.clear();
    };

    cl_uint num_devices = 0;
    for (auto platform : platforms) {

        num_devices = 0;
        ecode = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                0,
                nullptr,
                &num_devices
        );
        if (ecode != CL_SUCCESS || num_devices == 0) {
            ecode = CL_SUCCESS;
            //                clear_candidates();
            continue;
            //                RPY_HANDLE_OCL_ERROR(ecode);
        }

        devices.resize(num_devices);
        candidates.reserve(candidates.size() + num_devices);

        ecode = clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                num_devices,
                devices.data(),
                nullptr
        );
        if (ecode != CL_SUCCESS) {
            //                    clear_candidates();
            ecode = CL_SUCCESS;
            continue;
            //                    RPY_HANDLE_OCL_ERROR(ecode);
        }

        for (auto& dev : devices) {
            cl_device_type dev_type = 0;
            ecode = clGetDeviceInfo(
                    dev,
                    CL_DEVICE_TYPE,
                    sizeof(cl_device_type),
                    &dev_type,
                    nullptr
            );

            if (ecode != CL_SUCCESS) {
                ecode = CL_SUCCESS;
                continue;
            }

            if (dev_type == CL_DEVICE_TYPE_CPU) {
                candidates.push_back(dev);
                dev = nullptr;
            }
        }
        clear_devices();
    }

    if (!candidates.empty()) {
        // TODO: more sophisticated logic for choosing the best
        //  implementation of OpenCL to use. For now, just pick the first
        //  one.
        p_ocl_handle = new OCLDeviceHandle(candidates[0]);
        candidates[0] = nullptr;

        clear_candidates();
    }
}
CPUDeviceHandle::~CPUDeviceHandle() = default;

CPUDevice CPUDeviceHandle::get()
{
    static Rc<CPUDeviceHandle> device(new CPUDeviceHandle);
    return device;
}

DeviceInfo CPUDeviceHandle::info() const noexcept
{
    return {DeviceType::CPU, 0};
}

Buffer CPUDeviceHandle::raw_alloc(dimn_t count, dimn_t alignment) const
{
    if (alignment == 0) { alignment = alignof(std::max_align_t); }

    return Buffer(new CPUBuffer(
            mem::aligned_alloc(alignment, count),
            count
    ));
}
void CPUDeviceHandle::raw_free(void* pointer, dimn_t size) const
{
    mem::aligned_free(pointer);
}

template <typename... Args>
Kernel make_kernel(void (*fn)(Args...)) noexcept
{
    (void) fn;
    return Kernel();
}

static const bc::flat_map<string_view, Kernel> s_kernels{
        //        {"masked_uminus_double",
        //         make_kernel(kernels::masked_binary_into_buffer<
        //         double, std::plus<double>>)}
};

optional<Kernel> CPUDeviceHandle::get_kernel(const string& name) const noexcept
{
    auto kernel = DeviceHandle::get_kernel(name);
    if (kernel) { return kernel; }

    kernel = p_ocl_handle->get_kernel(name);
    if (kernel) { return kernel; }

    auto found = s_kernels.find(name);
    if (found != s_kernels.end()) { return {found->second}; }

    return {};
}
optional<Kernel>
CPUDeviceHandle::compile_kernel_from_str(const ExtensionSourceAndOptions& args
) const
{
    if (p_ocl_handle) { return p_ocl_handle->compile_kernel_from_str(args); }
    return {};
}
void CPUDeviceHandle::compile_kernels_from_src(
        const ExtensionSourceAndOptions& args
) const
{
    if (p_ocl_handle) { p_ocl_handle->compile_kernels_from_src(args); }
}
Event CPUDeviceHandle::new_event() const
{
    if (p_ocl_handle) { return p_ocl_handle->new_event(); }
    return Event(new CPUEvent);
}
Queue CPUDeviceHandle::new_queue() const
{
    if (p_ocl_handle) { return p_ocl_handle->new_queue(); }
    return Queue();
}
Queue CPUDeviceHandle::get_default_queue() const { return Queue(); }
bool CPUDeviceHandle::supports_type(const TypeInfo& info) const noexcept
{
    return true;
}
OCLDevice CPUDeviceHandle::ocl_device() const noexcept { return p_ocl_handle; }
DeviceCategory CPUDeviceHandle::category() const noexcept
{
    return DeviceCategory::CPU;
}
bool CPUDeviceHandle::has_compiler() const noexcept
{
    if (p_ocl_handle) { return p_ocl_handle->has_compiler(); }
    return false;
}
Device CPUDeviceHandle::compute_delegate() const
{
    if (!p_ocl_handle) {
        RPY_THROW(
                std::runtime_error,
                "no compute delegate is available on "
                "the host device"
        );
    }
    return p_ocl_handle;
}
