//
// Created by user on 23/08/23.
//

#ifndef ROUGHPY_PLATFORM_SRC_OPEN_CL_DEVICE_H_
#define ROUGHPY_PLATFORM_SRC_OPEN_CL_DEVICE_H_

#include <roughpy/platform/device.h>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#if defined(RPY_PLATFORM_MACOS)
#include <opencl/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <boost/container/flat_map.hpp>

#include <unordered_map>


namespace rpy {
namespace platform {



class OpenCLDevice : public DeviceHandle
{
    cl::Context m_context;
    cl::Platform m_platform;
    cl::CommandQueue m_queue;

    std::unordered_map<void*, int> m_allocations;

    boost::container::flat_map<string, cl::Kernel> m_kernels;

public:

    explicit OpenCLDevice(DeviceType type, int32_t device_id=0);

    void* raw_allocate(dimn_t size, dimn_t alignment) const override;
    void raw_dealloc(void* raw_pointer, dimn_t size) const override;

    void copy_to_device(void* d_dst_raw, const void* h_src_raw, dimn_t count)
            const override;
    void copy_from_device(void* h_dst_raw, const void* d_src_raw, dimn_t count)
            const override;
};

}// namespace platform
}// namespace rpy

#endif// ROUGHPY_PLATFORM_SRC_OPEN_CL_DEVICE_H_
