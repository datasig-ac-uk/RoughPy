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
    : public Memory
{
    mutable std::atomic<intptr_t> m_ref_count;
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

protected:
    void inc_ref() const noexcept override;
    RPY_NO_DISCARD bool dec_ref() const noexcept override;

public:
    ~HostAddressMemory() override;

    RPY_NO_DISCARD intptr_t ref_count() const noexcept override;
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
