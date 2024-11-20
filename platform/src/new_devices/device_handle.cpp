//
// Created by sam on 20/11/24.
//


#include "roughpy/devices/device_handle.h"


using namespace rpy;
using namespace rpy::devices;

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
Rc<const DeviceHandle> DeviceHandle::host() noexcept
{
    return nullptr;
}


