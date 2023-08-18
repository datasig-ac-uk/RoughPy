//
// Created by sam on 17/08/23.
//

#ifndef ROUGHPY_DEVICE_H
#define ROUGHPY_DEVICE_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "filesystem.h"

#if defined(__NVCC__)
#  include <cuda.h>

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE __inline__

#elif defined(__HIPCC__)

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE

#else
#  define RPY_DEVICE
#  define RPY_HOST
#  define RPY_DEVICE_HOST
#  define RPY_KERNEL
#  define RPY_DEVICE_SHARED
#  define RPY_STRONG_INLINE

#endif



namespace rpy { namespace platform {

using dindex_t = int;
using dsize_t = unsigned int;


/**
 * @brief Code for different device types
 *
 * These codes are chosen to be compatible with the DLPack
 * array interchange protocol. They enumerate the various different
 * device types that scalar data may be allocated on. This code goes
 * with a 32bit integer device ID, which is implementation specific.
 */
enum DeviceType : int32_t {
    CPU = 1,
    CUDA = 2,
    CUDAHost = 3,
    OpenCL = 4,
    Vulkan = 7,
    Metal = 8,
    VPI = 9,
    ROCM = 10,
    ROCMHost = 11,
    ExtDev = 12,
    CUDAManaged = 13,
    OneAPI = 14,
    WebGPU = 15,
    Hexagon = 16
};

/**
 * @brief Device type/id pair to identify a device
 *
 *
 */
struct DeviceInfo {
    DeviceType device_type;
    int32_t device_id;
};


class RPY_EXPORT DeviceHandle {
    DeviceInfo m_info;

public:

    virtual ~DeviceHandle();

    explicit DeviceHandle(DeviceInfo info)
        : m_info(info)
    {}

    explicit DeviceHandle(DeviceType type, int32_t device_id)
            : m_info {type, device_id}
    {}

    RPY_NO_DISCARD
    const DeviceInfo& info() const noexcept { return m_info; }

    RPY_NO_DISCARD
    virtual optional<fs::path> runtime_library() const noexcept = 0;


//    virtual void launch_kernel(const void* kernel,
//                               const void* launch_config,
//                               void** args
//                               ) = 0;


};


std::shared_ptr<DeviceHandle> get_cpu_device_handle() noexcept;



constexpr bool
operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

}}


#endif// ROUGHPY_DEVICE_H
