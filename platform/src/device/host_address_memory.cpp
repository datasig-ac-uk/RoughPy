//
// Created by sam on 21/11/24.
//


#include "roughpy/device/host_address_memory.h"

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/device/device_handle.h"

using namespace rpy;
using namespace rpy::device;

HostAddressMemory::HostAddressMemory(
        void* data,
        const generics::Type& type,
        const DeviceHandle& device,
        size_t size,
        size_t bytes,
        MemoryMode mode
) :
     Memory(type, device, size, bytes, mode),
     p_data(data)
{}
HostAddressMemory::~HostAddressMemory()
{
    if (RPY_LIKELY(p_data != nullptr)) { device().destroy_memory(*this); }
}

void HostAddressMemory::inc_ref() const noexcept
{

}
bool HostAddressMemory::dec_ref() const noexcept { return true; }
intptr_t HostAddressMemory::ref_count() const noexcept
{
    return 1;
}
const void* HostAddressMemory::data() const
{
    RPY_CHECK_NE(mode(), MemoryMode::WriteOnly);
    return p_data;
}

void* HostAddressMemory::data()
{
    RPY_CHECK_NE(mode(), MemoryMode::ReadOnly);
    return p_data;
}
bool HostAddressMemory::is_null() const noexcept { return p_data == nullptr; }

