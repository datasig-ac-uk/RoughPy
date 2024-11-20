//
// Created by sam on 20/11/24.
//

#ifndef ROUGHPY_GENERICS_MEMORY_H
#define ROUGHPY_GENERICS_MEMORY_H

#include <atomic>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/generics/type_ptr.h"

namespace rpy::devices {

// Forward declaration
class DeviceHandle;

enum class MemoryMode : uint8_t {
    ReadOnly,
    WriteOnly,
    ReadWrite
};


class ROUGHPY_PLATFORM_EXPORT Memory {
    mutable std::atomic_intptr_t m_ref_count;
    generics::TypePtr p_type;
    Rc<const DeviceHandle> p_device;
    size_t m_no_elements;
    size_t m_bytes;

protected:

    Memory(const generics::Type& type,
        const DeviceHandle& device,
        size_t no_elements,
        size_t bytes);

public:

    virtual ~Memory();

    RPY_NO_DISCARD virtual size_t size() const noexcept;

    RPY_NO_DISCARD MemoryMode mode() const noexcept;




};

} // rpy::generics

#endif //ROUGHPY_GENERICS_MEMORY_H
