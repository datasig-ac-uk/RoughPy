//
// Created by sam on 20/11/24.
//


#include "roughpy/device/device_handle.h"

#include "roughpy/device/host_address_memory.h"

#include <cstddef>

#include "roughpy/platform/alloc.h"

#include "roughpy/generics/type.h"

#include "host_device.h"


using namespace rpy;
using namespace rpy::device;

DeviceHandle::~DeviceHandle() = default;
DeviceHandle::DeviceHandle() noexcept : m_ref_count(0) {}
void DeviceHandle::inc_ref() const noexcept
{
    this->m_ref_count.fetch_add(1, std::memory_order_relaxed);
}
bool DeviceHandle::dec_ref() const noexcept
{
    auto old = this->m_ref_count.fetch_sub(1, std::memory_order_acq_rel);
    return old == 1;
}
intptr_t DeviceHandle::ref_count() const noexcept
{
    return this->m_ref_count.load(std::memory_order_acquire);
}

bool DeviceHandle::is_host() const noexcept
{
    return false;
}
Rc<const DeviceHandle> DeviceHandle::host() noexcept
{
    static const HostDeviceHandle host;
    return &host;
}




