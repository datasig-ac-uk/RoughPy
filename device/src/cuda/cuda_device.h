//
// Created by sam on 23/10/23.
//

#ifndef ROUGHPY_CUDA_DEVICE_H
#define ROUGHPY_CUDA_DEVICE_H

#include <roughpy/device/device_handle.h>

#include "cuda_headers.h"

namespace rpy {
namespace devices {

class CUDADeviceHandle : public DeviceHandle
{
    cudaDeviceProp m_properties;

    Buffer make_buffer(void* data, dimn_t size) const noexcept;
    Event make_event() const noexcept;
    Kernel make_kernel() const noexcept;
    Queue make_queue() const noexcept;

public:
    ~CUDADeviceHandle() override;
    DeviceType type() const noexcept override;
    DeviceCategory category() const noexcept override;
    DeviceInfo info() const noexcept override;
    optional<fs::path> runtime_library() const noexcept override;
    Buffer raw_alloc(dimn_t count, dimn_t alignment) const override;
    void raw_free(void* pointer, dimn_t size) const override;
    bool has_compiler() const noexcept override;
    const Kernel& register_kernel(Kernel kernel) const override;
    optional<Kernel> get_kernel(const string& name) const noexcept override;

    Event new_event() const override;
    Queue new_queue() const override;
    Queue get_default_queue() const override;
    optional<boost::uuids::uuid> uuid() const noexcept override;
    optional<PCIBusInfo> pci_bus_info() const noexcept override;
    bool supports_type(const TypeInfo& info) const noexcept override;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_CUDA_DEVICE_H
