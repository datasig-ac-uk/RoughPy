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
namespace devices {

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
    VendorID = 1,
    UUID = 2,
    PCI = 3
};

struct PCIBusInfo {
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
};

class DeviceSpecification
{
    DeviceCategory m_category;
    DeviceIdType m_id_type;

    union
    {
        uint32_t m_vendor_id;
        boost::uuids::uuid m_uuid;
        PCIBusInfo m_pci;
    };

    bool m_strict = false;

public:
    constexpr explicit DeviceSpecification(
            DeviceCategory cat,
            uint32_t vendor_id
    )
        : m_category(cat),
          m_id_type(DeviceIdType::VendorID),
          m_vendor_id(vendor_id)
    {}

    RPY_NO_DISCARD constexpr DeviceCategory category() const noexcept
    {
        return m_category;
    }

    RPY_NO_DISCARD constexpr bool is_strict() const noexcept
    {
        return m_strict;
    }

    inline void strict(bool strict) noexcept { m_strict = strict; }

    RPY_NO_DISCARD constexpr bool has_id() const noexcept
    {
        return m_id_type != DeviceIdType::None;
    }

    RPY_NO_DISCARD constexpr DeviceIdType id_type() const noexcept
    {
        return m_id_type;
    }

    RPY_NO_DISCARD constexpr const uint32_t& vendor_id() const noexcept
    {
        return m_vendor_id;
    }

    RPY_NO_DISCARD constexpr const boost::uuids::uuid& uuid() const noexcept
    {
        return m_uuid;
    }

    RPY_NO_DISCARD constexpr const PCIBusInfo& pci_addr() const noexcept
    {
        return m_pci;
    }
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

/**
 * @brief Type codes for different types.
 *
 * These are chosen to be compatible with the DLPack
 * array interchange protocol. Rational types will
 * be encoded as OpaqueHandle, since they're not simple
 * data. Some of these types might not be compatible with
 * this library.
 *
 * We have added some additional types that are available in RoughPy such as
 * rationals and polynomials.
 */
enum class TypeCode : uint8_t
{
    Int = 0U,
    UInt = 1U,
    Float = 2U,
    OpaqueHandle = 3U,
    BFloat = 4U,
    Complex = 5U,
    Bool = 6U,
    Rational = 7U,
    ArbitraryPrecision = 8U,
    ArbitraryPrecisionInt = ArbitraryPrecision | Int,
    ArbitraryPrecisionUInt = ArbitraryPrecision | UInt,
    ArbitraryPrecisionFloat = ArbitraryPrecision | Float,
    // ArbitraryPrecisionOpaque = ArbitraryPrecision | Opaque,
    // ArbitraryPrecisionBFloat = ArbitraryPrecision | BFloat,
    ArbitraryPrecisionComplex = ArbitraryPrecision | Complex,
    // ArbitraryPrecisionBool = ArbitraryPrecision | Bool,
    ArbitraryPrecisionRational = ArbitraryPrecision | Rational,

    Polynomial = 16U,
    APRationalPolynomial = Polynomial // | ArbitraryPrecisionRational

};

/**
 * @brief Basic information for identifying the type, size, and
 * configuration of a type.
 *
 * This was originally based on the DLPack protocol, but actually that proved
 * to be more effort converting bits to/from bytes. Now the size field is the
 * number of bytes rather than number of bits, since almost all scalar types
 * will be an integer number of bytes anyway.
 *
 * The lanes member will almost certainly be 1, and is currently ignored by
 * RoughPy.
 */
struct TypeInfo {
    TypeCode code;
    uint8_t bytes;
    uint16_t lanes = 1;
};

template <typename I>
struct BasicDim3 {
    I x;
    I y;
    I z;

    template <typename I1 = I, typename I2 = I, typename I3 = I>
    constexpr explicit BasicDim3(I1 i1 = 0, I2 i2 = 0, I3 i3 = 0)
        : x(i1),
          y(i2),
          z(i3)
    {}
};

using Dim3 = BasicDim3<dsize_t>;
using Size3 = BasicDim3<dimn_t>;

enum class BufferMode
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3
};

enum class EventStatus : int8_t
{
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
class HostDeviceHandle;
class KernalArgument;
class KernelInterface;
class Kernel;
class MemoryView;
class QueueInterface;
class Queue;

using Device = boost::intrusive_ptr<const DeviceHandle>;
using HostDevice = boost::intrusive_ptr<const HostDeviceHandle>;

RPY_EXPORT HostDevice get_host_device();
RPY_EXPORT Device get_default_device();
RPY_EXPORT optional<Device> get_device(const DeviceSpecification& spec);

constexpr bool operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}




namespace dtl {

template <typename T>
constexpr TypeInfo type_info();

template <typename T>
constexpr enable_if_t<is_integral<T>::value && is_signed<T>::value, TypeInfo>
type_info() {
    return { TypeCode::Int, sizeof(T), 1};
}

template <typename T>
constexpr enable_if_t<is_integral<T>::value && !is_signed<T>::value, TypeInfo>
type_info() {
    return { TypeCode::UInt, sizeof(T), 1};
}

template <typename T>
constexpr enable_if_t<is_floating_point<T>::value, TypeInfo>
type_info() {
    return { TypeCode::Float, sizeof(T), 1};
}

}

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
