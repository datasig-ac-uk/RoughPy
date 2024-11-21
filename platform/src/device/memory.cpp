//
// Created by sam on 20/11/24.
//

#include "roughpy/device/memory.h"

#include "roughpy/generics/type.h"



using namespace rpy;
using namespace rpy::device;

Memory::Memory(
        const generics::Type& type,
        const DeviceHandle& device,
        size_t no_elements,
        size_t bytes,
        MemoryMode mode
)
    : p_type(&type), p_device(&device), m_no_elements(no_elements), m_bytes(bytes), m_mode(mode)
{}
Memory::~Memory() = default;
size_t Memory::size() const noexcept
{
   return m_no_elements;
}
MemoryMode Memory::mode() const noexcept
{
    return m_mode;
}
const void* Memory::data() const
{
    RPY_THROW(std::runtime_error,
        "Direct memory access is not possible for this memory type");
}
void* Memory::data()
{
    RPY_THROW(
            std::runtime_error,
            "Direct memory access is not possible for this memory type"
    );
}


bool Memory::is_null() const noexcept
{
    return true;
}
bool Memory::empty() const noexcept
{
    return m_no_elements == 0;
}
