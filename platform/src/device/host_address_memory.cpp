//
// Created by sam on 21/11/24.
//

#include "roughpy/device/host_address_memory.h"

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/generics/type.h"

#include "roughpy/device/device_handle.h"

using namespace rpy;
using namespace rpy::device;

HostAddressMemory::HostAddressMemory(
        const generics::Type& type,
        const DeviceHandle& device,
        void* data,
        size_t size,
        size_t bytes,
        MemoryMode mode
)
    : p_type(&type),
      p_device(&device),
      p_data(data),
      m_no_elements(size),
      m_bytes(bytes),
      m_mode(mode)
{}
HostAddressMemory::~HostAddressMemory()
{
    if (RPY_LIKELY(p_data != nullptr && mode() != MemoryMode::ReadOnly)) {
        p_device->destroy_memory(*this);
    }
}

const generics::Type& HostAddressMemory::type() const noexcept
{
    return *p_type;
}
const DeviceHandle& HostAddressMemory::device() const noexcept
{
    return *p_device;
}
MemoryMode HostAddressMemory::mode() const noexcept
{
    return m_mode;
}
const void* HostAddressMemory::data() const
{
    RPY_CHECK_NE(m_mode, MemoryMode::WriteOnly);
    return p_data;
}

void* HostAddressMemory::data()
{
    RPY_CHECK_NE(m_mode, MemoryMode::ReadOnly);
    return p_data;
}


bool HostAddressMemory::is_null() const noexcept { return p_data == nullptr; }
size_t HostAddressMemory::size() const noexcept { return m_no_elements; }
size_t HostAddressMemory::bytes() const noexcept
{
    return m_bytes;
}
bool HostAddressMemory::empty() const noexcept { return m_no_elements == 0; }

MutableMemoryView HostAddressMemory::map_memory(size_t offset, size_t npos)
{
    RPY_CHECK_LT(offset, m_no_elements);

    if (npos == static_cast<size_t>(-1)) {
        npos = m_no_elements - offset;
    } else if (offset + npos > m_no_elements) {
        npos = m_no_elements - offset;
    }

    const auto byte_offset = offset * type().object_size();
    byte* ptr = static_cast<byte*>(p_data) + byte_offset;

    return { this, ptr, npos };
}
ConstMemoryView
HostAddressMemory::map_const_memory(size_t offset, size_t npos) const
{
    RPY_CHECK_LT(offset, m_no_elements);

    if (npos == static_cast<size_t>(-1)) {
        npos = m_no_elements - offset;
    } else if (offset + npos > m_no_elements) {
        npos = m_no_elements - offset;
    }

    const auto byte_offset = offset * type().object_size();
    const byte* ptr = static_cast<const byte*>(p_data) + byte_offset;
    return { this, ptr, npos };
}

void HostAddressMemory::unmap(const ConstMemoryView& child) const
{
    // Nothing to do
}

void HostAddressMemory::unmap(const MutableMemoryView& child)
{
    // Nothing to do
}
