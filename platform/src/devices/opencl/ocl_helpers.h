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

#ifndef ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_
#define ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_

#include "ocl_handle_errors.h"
#include "ocl_headers.h"

#include <roughpy/core/debug_assertion.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>
#include "devices/core.h"

namespace rpy {
namespace devices {
namespace cl {

template <typename Fn, typename CLObj, typename Info>
RPY_NO_DISCARD inline string
string_info(Fn&& fn, CLObj* cl_object, Info info_id)
{
    size_t ret_size;
    auto ecode = fn(cl_object, info_id, 0, nullptr, &ret_size);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    string result;
    result.resize(ret_size);
    ecode = fn(cl_object, info_id, result.size(), result.data(), nullptr);
    if (ecode != CL_SUCCESS) { RPY_HANDLE_OCL_ERROR(ecode); }

    if (result.back() == '\0') {
        result.resize(ret_size - 1);
    }
    return result;
}

RPY_NO_DISCARD cl_mem
to_ocl_buffer(void* data, dimn_t size, cl_context context);

RPY_NO_DISCARD Slice<byte> from_ocl_buffer(cl_mem buf);

RPY_NO_DISCARD inline cl_device_type to_ocl_device_type(DeviceCategory category
) noexcept
{
    switch (category) {
        case DeviceCategory::CPU: return CL_DEVICE_TYPE_CPU;
        case DeviceCategory::GPU: return CL_DEVICE_TYPE_GPU;
        case DeviceCategory::FPGA: return CL_DEVICE_TYPE_ACCELERATOR;
        case DeviceCategory::DSP: return CL_DEVICE_TYPE_ACCELERATOR;
        case DeviceCategory::AIP: return CL_DEVICE_TYPE_ACCELERATOR;
        case DeviceCategory::Other: return CL_DEVICE_TYPE_CUSTOM;
    }
    RPY_UNREACHABLE_RETURN(CL_DEVICE_TYPE_CPU);
}

namespace dtl {

template <typename CLType>
class ScopedCLObject
{
    CLType m_data;

    using function_t = cl_int (*)(CLType);

    function_t p_retain;
    function_t p_release;

public:
    constexpr ScopedCLObject(
            CLType data,
            function_t retain,
            function_t release
    ) noexcept
        : m_data(data),
          p_retain(retain),
          p_release(release)
    {
        if (p_retain) { p_retain(m_data); }
    }

    ~ScopedCLObject() noexcept
    {
        if (m_data && p_release) {
            auto ecode = p_release(m_data);
            RPY_DBG_ASSERT(ecode == CL_SUCCESS);
        }
    }

    constexpr operator CLType() const noexcept { return m_data; }

    constexpr CLType* ptr() noexcept { return &m_data; }
    constexpr const CLType* ptr() const noexcept { return &m_data; }

};

}// namespace dtl

inline dtl::ScopedCLObject<cl_context> scoped_guard(cl_context data) noexcept
{
    return {data, clRetainContext, clReleaseContext};
}
inline dtl::ScopedCLObject<cl_device_id> scoped_guard(cl_device_id data
) noexcept
{
    return {data, clRetainDevice, clReleaseDevice};
}
inline dtl::ScopedCLObject<cl_program> scoped_guard(cl_program data) noexcept
{
    return {data, clRetainProgram, clReleaseProgram};
}
inline dtl::ScopedCLObject<cl_mem> scoped_guard(cl_mem data) noexcept
{
    return {data, clRetainMemObject, clReleaseMemObject};
}
inline dtl::ScopedCLObject<cl_command_queue> scoped_guard(cl_command_queue data
) noexcept
{
    return {data, clRetainCommandQueue, clReleaseCommandQueue};
}
inline dtl::ScopedCLObject<cl_kernel> scoped_guard(cl_kernel data) noexcept
{
    return {data, clRetainKernel, clReleaseKernel};
}
inline dtl::ScopedCLObject<cl_event> scoped_guard(cl_event data) noexcept
{
    return {data, clRetainEvent, clReleaseEvent};
}
inline dtl::ScopedCLObject<cl_sampler> scoped_guard(cl_sampler data) noexcept
{
    return {data, clRetainSampler, clReleaseSampler};
}

}// namespace cl
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_OPENCL_OCL_HELPERS_H_
