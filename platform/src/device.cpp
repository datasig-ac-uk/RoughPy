//
// Created by sam on 17/08/23.
//

#include <roughpy/platform/device.h>

#include "cpu_device/CPUDevice.h"

using namespace rpy;
using namespace rpy::platform;

std::shared_ptr<DeviceHandle> get_cpu_device_handle() noexcept
{
    static std::shared_ptr<DeviceHandle> cpu_handle(new CPUDevice);
    return cpu_handle;
}

DeviceHandle::~DeviceHandle() = default;