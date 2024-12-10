//
// Created by sam on 21/11/24.
//

#ifndef ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H
#define ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H


#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

#include "memory.h"

namespace rpy::device {

class ROUGHPY_PLATFORM_EXPORT HostAddressMemory
    : public mem::RefCountedMiddle<Memory>
{
    generics::TypePtr p_type;
    Rc<const DeviceHandle> p_device;
    void* p_data;
    size_t m_no_elements;
    size_t m_bytes;
    MemoryMode m_mode;

public:
    HostAddressMemory(
            const generics::Type& type,
            const DeviceHandle& device,
            void* data,
            size_t size,
            size_t bytes,
            MemoryMode mode=MemoryMode::ReadWrite
    );


public:
    ~HostAddressMemory() override;

    const generics::Type& type() const noexcept override;
    RPY_NO_DISCARD const DeviceHandle& device() const noexcept override;
    RPY_NO_DISCARD MemoryMode mode() const noexcept override;

    const void* data() const override;
    void* data() override;
    RPY_NO_DISCARD bool is_null() const noexcept override;

    void* release() noexcept { return std::exchange(p_data, nullptr); }

    RPY_NO_DISCARD size_t size() const noexcept override;
    RPY_NO_DISCARD size_t bytes() const noexcept override;
    RPY_NO_DISCARD bool empty() const noexcept override;
    RPY_NO_DISCARD MutableMemoryView
    map_memory(size_t offset, size_t npos) override;
    RPY_NO_DISCARD ConstMemoryView
    map_const_memory(size_t offset, size_t npos) const override;

protected:
    void unmap(const ConstMemoryView& child) const override;
    void unmap(const MutableMemoryView& child) override;
};

}// namespace rpy::device
#endif //ROUGHPY_DEVICE_HOST_ADDRESS_MEMORY_H
