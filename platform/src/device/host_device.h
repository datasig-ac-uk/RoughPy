//
// Created by sam on 21/11/24.
//

#ifndef ROUGHPY_DEVICE_INTERNAL_HOST_DEVICE_H
#define ROUGHPY_DEVICE_INTERNAL_HOST_DEVICE_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/device/device_handle.h"


namespace rpy {
namespace device {

class HostDeviceHandle : public DeviceHandle {

protected:
    void inc_ref() const noexcept override;
    RPY_NO_DISCARD bool dec_ref() const noexcept override;

public:
    RPY_NO_DISCARD intptr_t ref_count() const noexcept override;

    bool is_host() const noexcept override;
    bool supports_type(const generics::Type& type) const noexcept override;
    RPY_NO_DISCARD Rc<Memory> allocate_memory(
            const generics::Type& type,
            size_t size,
            size_t alignment
    ) const override;
    void copy_memory(Memory& dst, const Memory& src) const override;
    void destroy_memory(Memory& memory) const override;
};

} // device
} // rpy

#endif //ROUGHPY_DEVICE_INTERNAL_HOST_DEVICE_H
