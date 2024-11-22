//
// Created by sam on 20/11/24.
//

#ifndef ROUGHPY_PLATFORM_DEVICE_HANDLE_H
#define ROUGHPY_PLATFORM_DEVICE_HANDLE_H

#include <atomic>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/smart_ptr.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/alloc.h"
#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy {

namespace generics {

class Type;

}

namespace device {

class Memory;

class ROUGHPY_PLATFORM_EXPORT DeviceHandle : public mem::PolymorphicRefCounted
{

protected:
    DeviceHandle() noexcept;

public:

    virtual bool is_host() const noexcept;
    virtual bool supports_type(const generics::Type& type) const noexcept = 0;

    RPY_NO_DISCARD virtual Rc<Memory> allocate_memory(
            const generics::Type& type,
            size_t size,
            size_t alignment
    ) const = 0;

    virtual void copy_memory(Memory& dst, const Memory& src) const = 0;

    virtual void destroy_memory(Memory& memory) const = 0;

    static Rc<const DeviceHandle> host() noexcept;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_PLATFORM_DEVICE_HANDLE_H
