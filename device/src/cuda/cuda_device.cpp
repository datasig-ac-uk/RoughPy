//
// Created by sam on 23/10/23.
//

#include "cuda_device.h"
#include "cuda_buffer.h"
#include "cuda_event.h"
#include "cuda_kernel.h"
#include "cuda_queue.h"

#include <boost/uuid/uuid.hpp>

using namespace rpy;
using namespace rpy::devices;

CUDADeviceHandle::~CUDADeviceHandle() {}
DeviceType CUDADeviceHandle::type() const noexcept { return DeviceType::CUDA; }
DeviceCategory CUDADeviceHandle::category() const noexcept
{
    return DeviceCategory::GPU;
}

DeviceInfo CUDADeviceHandle::info() const noexcept
{
    return {DeviceType::CUDA, 0};
}
optional<fs::path> CUDADeviceHandle::runtime_library() const noexcept
{
    return DeviceHandle::runtime_library();
}
Buffer CUDADeviceHandle::raw_alloc(dimn_t count, dimn_t alignment) const
{
    void* dp_data;
    auto ecode = cudaMalloc(&dp_data, count);
    RPY_CHECK(ecode == cudaSuccess);
    return make_buffer(dp_data, count);
}
void CUDADeviceHandle::raw_free(void* pointer, dimn_t size) const
{
    auto ecode = cudaFree(pointer);
    RPY_CHECK(ecode == cudaSuccess);
}
bool CUDADeviceHandle::has_compiler() const noexcept
{
    return DeviceHandle::has_compiler();
}
const Kernel& CUDADeviceHandle::register_kernel(Kernel kernel) const
{
    return DeviceHandle::register_kernel(kernel);
}
optional<Kernel> CUDADeviceHandle::get_kernel(const string& name) const noexcept
{
    return DeviceHandle::get_kernel(name);
}

Event CUDADeviceHandle::new_event() const { return DeviceHandle::new_event(); }
Queue CUDADeviceHandle::new_queue() const { return DeviceHandle::new_queue(); }
Queue CUDADeviceHandle::get_default_queue() const
{
    return DeviceHandle::get_default_queue();
}
optional<boost::uuids::uuid> CUDADeviceHandle::uuid() const noexcept
{
    boost::uuids::uuid uuid;
    std::memcpy(uuid.data, m_properties.uuid.bytes, sizeof(uuid));

    return uuid;
}
optional<PCIBusInfo> CUDADeviceHandle::pci_bus_info() const noexcept
{
    return {
            {static_cast<uint32_t>(m_properties.pciDomainID),
             static_cast<uint32_t>(m_properties.pciBusID),
             static_cast<uint32_t>(m_properties.pciDeviceID),
             0}
    };
}
bool CUDADeviceHandle::supports_type(const TypeInfo& info) const noexcept
{
    return DeviceHandle::supports_type(info);
}