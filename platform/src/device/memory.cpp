//
// Created by sam on 20/11/24.
//

#include "roughpy/device/memory.h"

#include "roughpy/core/smart_ptr.h"

#include "roughpy/generics/type.h"

#include "roughpy/device/device_handle.h"
#include "roughpy/device/host_address_memory.h"

using namespace rpy;
using namespace rpy::device;

Memory::Memory(
        const generics::Type& type,
        const DeviceHandle& device,
        size_t no_elements,
        size_t bytes,
        MemoryMode mode
)
    : p_type(&type),
      p_device(&device),
      m_no_elements(no_elements),
      m_bytes(bytes),
      m_mode(mode)
{}
Memory::~Memory() {};
size_t Memory::size() const noexcept { return m_no_elements; }
MemoryMode Memory::mode() const noexcept { return m_mode; }
const void* Memory::data() const
{
    RPY_THROW(
            std::runtime_error,
            "Direct memory access is not possible for this memory type"
    );
}
void* Memory::data()
{
    RPY_THROW(
            std::runtime_error,
            "Direct memory access is not possible for this memory type"
    );
}

bool Memory::is_null() const noexcept { return true; }
bool Memory::empty() const noexcept { return m_no_elements == 0; }

Rc<Memory> Memory::to(const DeviceHandle& device) const
{
    const auto& this_type = this->type();
    RPY_CHECK(device.supports_type(this_type));
    auto buffer = device.allocate_memory(
            this_type,
            size(),
            alignof(std::max_align_t)
    );
    device.copy_memory(*buffer, *this);
    return buffer;
}
Rc<Memory> Memory::to_host() const { return to(*DeviceHandle::host()); }

