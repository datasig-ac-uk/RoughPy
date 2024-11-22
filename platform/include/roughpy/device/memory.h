//
// Created by sam on 20/11/24.
//

#ifndef ROUGHPY_DEVICE_MEMORY_H
#define ROUGHPY_DEVICE_MEMORY_H

#include <atomic>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/generics/type_ptr.h"

namespace rpy::device {

// Forward declaration
class DeviceHandle;

enum class MemoryMode : uint8_t
{
    ReadOnly,
    WriteOnly,
    ReadWrite
};

class ROUGHPY_PLATFORM_EXPORT Memory : public mem::PolymorphicRefCounted
{
    generics::TypePtr p_type;
    Rc<const DeviceHandle> p_device;
    size_t m_no_elements;
    size_t m_bytes;
    MemoryMode m_mode;

protected:
    Memory(const generics::Type& type,
           const DeviceHandle& device,
           size_t no_elements,
           size_t bytes,
           MemoryMode mode);

public:

    RPY_NO_DISCARD virtual size_t size() const noexcept;
    RPY_NO_DISCARD size_t bytes() const noexcept { return m_bytes; }
    RPY_NO_DISCARD const generics::Type& type() const noexcept
    {
        return *p_type;
    }

    RPY_NO_DISCARD MemoryMode mode() const noexcept;

    virtual const void* data() const;
    virtual void* data();

    const DeviceHandle& device() const noexcept { return *p_device; }
    RPY_NO_DISCARD virtual bool is_null() const noexcept;
    RPY_NO_DISCARD virtual bool empty() const noexcept;

    RPY_NO_DISCARD Rc<Memory> to(const DeviceHandle& device) const;
    RPY_NO_DISCARD Rc<Memory> to_host() const;


};

}// namespace rpy::device

#endif// ROUGHPY_DEVICE_MEMORY_H
