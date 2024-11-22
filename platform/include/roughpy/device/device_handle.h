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

enum class DeviceCategory : int32_t
{
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    DSP = 3,
    AIP = 4,
    Other = 5
};

enum class DeviceIdType : int32_t
{
    None = 0,
    VendorID = 1,
    UUID = 2,
    PCI = 3
};


// Forward declarations of types that the device handle needs to interact with
class Memory;
class Queue;
class Event;

/**
 * @brief Represents a handle to a device, which abstracts the interaction with
 * hardware or virtual devices.
 *
 * This class provides methods to open, close, read from, write to, and manage
 * the state of a device. It encapsulates the details of device communication
 * and ensures safe and controlled access to the device.
 *
 * Typical usage involves creating an instance of DeviceHandle, opening a
 * connection to the device, performing read/write operations, and finally
 * closing the connection.
 */
class ROUGHPY_PLATFORM_EXPORT DeviceHandle : public mem::PolymorphicRefCounted
{

protected:
    DeviceHandle() noexcept;

public:

    virtual bool is_host() const noexcept;
    virtual bool supports_type(const generics::Type& type) const noexcept = 0;

    // Memory management
    RPY_NO_DISCARD virtual Rc<Memory> allocate_memory(
            const generics::Type& type,
            size_t size,
            size_t alignment
    ) const = 0;
    virtual void copy_memory(Memory& dst, const Memory& src) const = 0;
    virtual void destroy_memory(Memory& memory) const = 0;

    // Event management
    virtual Rc<Event> new_event() const;

    // Queue management
    virtual Rc<Queue> new_queue() const;


    static Rc<const DeviceHandle> host() noexcept;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_PLATFORM_DEVICE_HANDLE_H
