//
// Created by sam on 20/11/24.
//


#include "roughpy/device/device_handle.h"


#include <cstddef>

#include "roughpy/core/smart_ptr.h"

#include "roughpy/platform/alloc.h"

#include "roughpy/generics/type.h"

#include "roughpy/device/host_address_memory.h"

#include "host_device.h"


using namespace rpy;
using namespace rpy::device;

DeviceHandle::DeviceHandle() noexcept  {}

bool DeviceHandle::is_host() const noexcept
{
    return false;
}
Rc<const DeviceHandle> DeviceHandle::host() noexcept
{
    static const HostDeviceHandle host;
    return &host;
}




