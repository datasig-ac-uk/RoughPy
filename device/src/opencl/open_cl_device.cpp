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
// Created by sam on 25/08/23.
//

#include "open_cl_device.h"

#include <algorithm>
#include <mutex>
#include <vector>

static std::vector<cl_device_id> s_device_list;
static std::mutex s_device_lock;

static void set_device_id(cl_device_id device, int32_t& id) {
    std::lock_guard<std::mutex> access(s_device_lock);

    if (s_device_list.empty()) {
        id = 0;
        s_device_list.push_back(device);
    } else {
        auto begin = s_device_list.begin();
        auto end = s_device_list.end();
        auto pos = std::find(begin, end, cl_device_id());
        id = static_cast<int32_t>(pos - begin);

        if (pos != end) {
            *pos = device;
        } else {
            s_device_list.push_back(device);
        }
    }
}


rpy::device::OpenCLDevice::OpenCLDevice(
        const rpy::device::OpenCLRuntimeLibrary* rt_lib, cl_device_id device
)
    : DeviceHandle(rt_lib->buffer_interface(), rt_lib->event_interface(),
                   rt_lib->kernel_interface(), rt_lib->queue_interface()),
      p_runtime(rt_lib),
      m_device(device)
{
      set_device_id(m_device, m_device_id);
      cl_int errcode = CL_SUCCESS;

      m_ctx = p_runtime->clCreateContext(
              nullptr,
              1,
              &m_device,
              nullptr,
              nullptr,
              &errcode
              );
      RPY_CHECK(errcode == CL_SUCCESS);

}

rpy::device::OpenCLDevice::~OpenCLDevice() {
    std::lock_guard<std::mutex> access(s_device_lock);
    RPY_CHECK(s_device_list.size() > static_cast<dimn_t>(m_device_id));
    s_device_list[m_device_id] = cl_device_id();
}

rpy::device::DeviceInfo rpy::device::OpenCLDevice::info() const noexcept
{
    return { DeviceType::OpenCL, m_device_id };
}

std::optional<std::filesystem::path>
rpy::device::OpenCLDevice::runtime_library() const noexcept
{
    return p_runtime->location();
}
