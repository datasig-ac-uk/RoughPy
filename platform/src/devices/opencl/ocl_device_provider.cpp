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
// Created by user on 17/10/23.
//

#include "ocl_device_provider.h"

#include "roughpy/core/debug_assertion.h"    // for RPY_DBG_ASSERT

#include "ocl_device.h"
#include "ocl_handle_errors.h"
#include "ocl_headers.h"
#include "ocl_helpers.h"

#include "devices/device_handle.h"

#include <CL/cl_ext.h>
#include <boost/container/small_vector.hpp>

using namespace rpy;
using namespace rpy::devices;
namespace bc = boost::container;

bool OCLDeviceProvider::supports(DeviceCategory category) const noexcept
{
    if (category == DeviceCategory::CPU) { return false; }

    std::vector<cl_platform_id> platforms;
    cl_uint count = 0;
    auto ecode = clGetPlatformIDs(0, nullptr, &count);
    if (ecode != CL_SUCCESS || count == 0) { return false; }

    platforms.resize(count);
    ecode = clGetPlatformIDs(count, platforms.data(), nullptr);
    if (ecode != CL_SUCCESS) { return false; }

    auto cl_category = cl::to_ocl_device_type(category);

    count = 0;
    for (auto&& platform : platforms) {
        ecode = clGetDeviceIDs(platform, cl_category, 0, nullptr, &count);
        if (ecode == CL_SUCCESS && count > 0) { return true; }
    }

    return false;
}
int OCLDeviceProvider::priority(const DeviceSpecification& spec) const noexcept
{
    return 0;
}

namespace {

bool check_device_spec(
        cl_device_id device,
        const DeviceSpecification& spec
) noexcept
{
    // TODO: Handle non-strict
    switch (spec.id_type()) {
        case DeviceIdType::None: return true;
        case DeviceIdType::VendorID: {
            const auto& vid = spec.vendor_id();

            cl_uint d_vid = 0;
            auto ecode = clGetDeviceInfo(
                    device,
                    CL_DEVICE_VENDOR_ID,
                    sizeof(d_vid),
                    &d_vid,
                    nullptr
            );

            if (ecode != CL_SUCCESS) { return false; }

            if (spec.is_strict() && d_vid != vid) { return false; }

            break;
        }
        case DeviceIdType::UUID: {
            const auto& uuid = spec.uuid();

            boost::uuids::uuid d_uuid;
            auto ecode = clGetDeviceInfo(
                    device,
                    CL_DEVICE_UUID_KHR,
                    sizeof(uuid),
                    d_uuid.data,
                    nullptr
            );

            if (ecode != CL_SUCCESS) { return false; }

            if (spec.is_strict() && uuid != d_uuid) { return false; }

            break;
        }
        case DeviceIdType::PCI: {
            const auto& pci_addr = spec.pci_addr();

            cl_device_pci_bus_info_khr pci_bus;
            auto ecode = clGetDeviceInfo(
                    device,
                    CL_DEVICE_PCI_BUS_INFO_KHR,
                    sizeof(pci_bus),
                    &pci_bus,
                    nullptr
            );

            if (ecode != CL_SUCCESS) { return false; }

            if (spec.is_strict()) {

                if (pci_bus.pci_bus != pci_addr.pci_bus) {
                    return false;
                }
                if (pci_bus.pci_device != pci_addr.pci_device) {
                    return false;
                }
                if (pci_bus.pci_domain != pci_addr.pci_domain) {
                    return false;
                }
                if (pci_bus.pci_function != pci_addr.pci_function) {
                    return false;
                }
            }

            break;
        }
    }

    return true;
}

}// namespace

Device OCLDeviceProvider::get(const DeviceSpecification& specification) noexcept
{
    RPY_DBG_ASSERT(specification.category() != DeviceCategory::CPU);

    auto cl_category = cl::to_ocl_device_type(specification.category());

    bc::small_vector<cl_platform_id, 1> platforms;
    cl_uint count = 0;
    auto ecode = clGetPlatformIDs(0, nullptr, &count);
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);

    platforms.resize(count);
    ecode = clGetPlatformIDs(count, platforms.data(), nullptr);
    RPY_DBG_ASSERT(ecode == CL_SUCCESS);

    bc::small_vector<cl_device_id, 1> devices;
    bc::small_vector<cl_device_id, 1> candidates;

    auto clear_candidates = [&candidates]() {
        for (auto&& cand : candidates) { clReleaseDevice(cand); }
        candidates.clear();
    };

    for (auto&& platform : platforms) {
        count = 0;
        ecode = clGetDeviceIDs(platform, cl_category, 0, nullptr, &count);
        if (ecode != CL_SUCCESS || count == 0) { continue; }

        devices.resize(count);
        ecode = clGetDeviceIDs(
                platform,
                cl_category,
                count,
                devices.data(),
                nullptr
        );
        RPY_DBG_ASSERT(ecode == CL_SUCCESS);

        for (auto&& device : devices) {
            if (check_device_spec(device, specification)) {
                candidates.push_back(device);
            } else {
                clReleaseDevice(device);
            }
        }
        devices.clear();
    }

    if (!candidates.empty()) {
        // TODO: Better logic for picking best one
        auto device = candidates[0];
        candidates[0] = nullptr;
        clear_candidates();
        return new OCLDeviceHandle(device);
    }

    return nullptr;
}
