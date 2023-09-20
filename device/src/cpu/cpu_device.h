//
// Created by sam on 20/09/23.
//

#ifndef ROUGHPY_CPU_DEVICE_H
#define ROUGHPY_CPU_DEVICE_H

#include <roughpy/device/core.h>
#include <roughpy/device/buffer.h>
#include <roughpy/device/device_handle.h>
#include <roughpy/device/event.h>
#include <roughpy/device/kernel.h>
#include <roughpy/device/queue.h>

#include <boost/smart_ptr/intrusive_ptr.hpp>

#include <unordered_map>
#include <mutex>

namespace rpy {
namespace device {

class OpenCLDevice;

class CPUDevice : public DeviceHandle
{
    boost::intrusive_ptr<const OpenCLDevice> p_ocl_device;

    mutable std::recursive_mutex m_lock;

public:
    struct AllocationMetadata {
        dimn_t alignment;
        dimn_t count;
    };

private:
    mutable std::unordered_map<void*, AllocationMetadata> m_alloc_md;

public:

    static boost::intrusive_ptr<const CPUDevice> get() noexcept;

    const AllocationMetadata& get_metadata(void* ptr) const;


    ~CPUDevice() override;
    DeviceInfo info() const noexcept override;
    optional<fs::path> runtime_library() const noexcept override;
    Buffer raw_alloc(dimn_t count, dimn_t alignment) const override ;
    void raw_free(Buffer buffer) const override;

};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_CPU_DEVICE_H
