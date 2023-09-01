// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_CORE_H_
#define ROUGHPY_DEVICE_CORE_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>


#include "macros.h"


namespace rpy {
namespace device {


using dindex_t = int;
using dsize_t = unsigned int;


enum class DeviceCategory : int32_t  {
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    DSP = 3,
    AIP = 4,
    Other = 5
};


/**
 * @brief Code for different device types
 *
 * These codes are chosen to be compatible with the DLPack
 * array interchange protocol. They enumerate the various different
 * device types that scalar data may be allocated on. This code goes
 * with a 32bit integer device ID, which is implementation specific.
 */
enum DeviceType : int32_t
{
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


struct Dim3 {
    dsize_t x;
    dsize_t y;
    dsize_t z;

    template <typename I1=dsize_t, typename I2=dsize_t, typename I3=dsize_t>
    explicit Dim3(I1 i1=1, I2 i2=1, I3 i3=1)
        : x(i1), y(i2), z(i3)
    {}

};


struct KernelLaunchParams {
    Dim3 grid_work_size;
    Dim3 block_work_size;
    optional<Dim3> offsets_size;
    optional<dsize_t> dynamic_shared_memory;
};


enum class EventStatus : int8_t {
    CompletedSuccessfully = 0,
    Queued = 1,
    Submitted = 2,
    Running = 4,
    Error = 8
};


class RPY_EXPORT BufferInterface;
class RPY_EXPORT EventInterface;
class RPY_EXPORT KernelInterface;
class RPY_EXPORT QueueInterface;


class RPY_EXPORT DeviceHandle;
class RPY_EXPORT Kernel;
class RPY_EXPORT Queue;



RPY_NO_DISCARD RPY_EXPORT
std::shared_ptr<DeviceHandle> get_device(DeviceType device_type,
                                         int32_t device_id);


constexpr hash_t hash_value(const DeviceInfo& info) noexcept {
    return (static_cast<hash_t>(info.device_type) << (sizeof(hash_t) / 2))
            | static_cast<hash_t>(info.device_id);
}


}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
