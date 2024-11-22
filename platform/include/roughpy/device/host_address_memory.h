//
// Created by sam on 21/11/24.
//

#ifndef ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H
#define ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H

#include <atomic>


#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

#include "memory.h"

namespace rpy::device {

class ROUGHPY_PLATFORM_EXPORT HostAddressMemory
    : public mem::RefCountedMiddle<Memory>
{
    void* p_data;

public:
    HostAddressMemory(
            void* data,
            const generics::Type& type,
            const DeviceHandle& device,
            size_t size,
            size_t bytes,
            MemoryMode mode=MemoryMode::ReadWrite
    );


public:
    ~HostAddressMemory() override;

    const void* data() const override;
    void* data() override;
    RPY_NO_DISCARD bool is_null() const noexcept override;

    void* release() noexcept
    {
        return std::exchange(p_data, nullptr);
    }
};

}// namespace rpy::device
#endif //ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H
