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

#include "devices/core.h"
#include "devices/device_handle.h"
#include "devices/device_provider.h"

#include "host/host_device_impl.h"
#include "opencl/ocl_device_provider.h"

#include <boost/container/small_vector.hpp>

#include <mutex>

using namespace rpy;
using namespace rpy::devices;

static std::mutex s_provider_lock;
static boost::container::small_vector<std::unique_ptr<DeviceProvider>, 2>
        s_provider_list;

void DeviceProvider::register_provider(
        std::unique_ptr<DeviceProvider>&& provider
)
{
    std::lock_guard<std::mutex> access(s_provider_lock);
    if (s_provider_list.empty()) {
        s_provider_list.emplace_back(new OCLDeviceProvider);
    }

    s_provider_list.emplace_back(std::move(provider));
}

optional<Device> rpy::devices::get_device(const DeviceSpecification& spec)
{
    std::lock_guard<std::mutex> access(s_provider_lock);
    if (s_provider_list.empty()) {
        s_provider_list.emplace_back(new OCLDeviceProvider);
    }


    boost::container::small_vector<DeviceProvider*, 1> candidates;
    for (auto&& provider : s_provider_list) {
        if (provider->supports(spec.category())) {
            candidates.push_back(provider.get());
        }
    }

    if (!candidates.empty()) {
        return candidates[0]->get(spec);
    }

    return {};
}

HostDevice rpy::devices::get_host_device() { return CPUDeviceHandle::get(); }
Device rpy::devices::get_default_device() { return CPUDeviceHandle::get(); }
