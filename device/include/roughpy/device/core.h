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
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/uuid/uuid.hpp>

namespace rpy {
namespace device {

using dindex_t = int;
using dsize_t = unsigned int;

enum class DeviceCategory : int32_t
{
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    DSP = 3,
    AIP = 4,
    Other = 5
};

enum class DeviceIdType : int32_t
{
    None = 0,
    UUID = 1,
    PCI = 2
};

struct PCIBusInfo {
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
};

struct DeviceSpecification {
    DeviceCategory category;
    DeviceIdType id_type;

    union
    {
        boost::uuids::uuid uuid;
        PCIBusInfo pci;
    };
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

enum class EventStatus : int8_t {
    CompletedSuccessfully = 0,
    Queued = 1,
    Submitted = 2,
    Running = 4,
    Error = 8
};




class DeviceHandle;

class BufferInterface;
class Buffer;
class EventInterface;
class Event;
class KernelInterface;
class Kernel;
class QueueInterface;
class Queue;



using Device = boost::intrusive_ptr<const DeviceHandle>;


RPY_EXPORT Device get_default_device();
RPY_EXPORT Device get_device(const DeviceSpecification& spec);


constexpr bool operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}
}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
