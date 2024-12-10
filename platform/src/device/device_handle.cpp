//
// Created by sam on 20/11/24.
//


#include "roughpy/device/device_handle.h"


#include <cstddef>

#include "roughpy/core/smart_ptr.h"

#include "roughpy/platform/alloc.h"
#include "roughpy/platform/reference_counting.h"

#include "roughpy/generics/type.h"

#include "roughpy/device/event.h"
#include "roughpy/device/host_address_memory.h"
#include "roughpy/device/queue.h"

#include "host_device.h"


using namespace rpy;
using namespace rpy::device;

DeviceHandle::DeviceHandle() noexcept = default;

bool DeviceHandle::is_host() const noexcept
{
    return false;
}
Rc<const DeviceHandle> DeviceHandle::host() noexcept
{
    static const HostDeviceHandle host;
    return &host;
}

Rc<Event> DeviceHandle::new_event() const
{
    return Rc<Event>(new Event());
}
Rc<Queue> DeviceHandle::new_queue() const
{
    return Rc<Queue>(new Queue());
}
