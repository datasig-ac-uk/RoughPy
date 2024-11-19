// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_CORE_H_
#define ROUGHPY_DEVICE_CORE_H_

#include <iosfwd>


#include <roughpy/core/macros.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include "roughpy/core/smart_ptr.h"

#include <roughpy/platform/serialization.h>

#include <boost/uuid/uuid.hpp>

#include "rational_numbers.h"

#include "roughpy/platform/roughpy_platform_export.h"

/*
 * We use the half precision floating point and bfloat16 types from Eigen but
 * we really don't want to include the whole Eigen/Core header until we
 * absolutely have to. To avoid this, we pre-declare the two types that we need
 * in the Eigen namespace, so we can typedef them in the our own namespace and
 * set up all the machinery that we need. Then, we can import the actual
 * definitions only when we need to.
 */
namespace Eigen {

struct half;
struct bfloat16;

}// namespace Eigen



namespace rpy {
namespace devices {

using dindex_t = int;
using dsize_t = unsigned int;

/// IEEE half-precision floating point type
using Eigen::half;
/// BFloat16 (truncated) floating point type
using Eigen::bfloat16;
/// Rational scalar type
//using rational_scalar_type = lal::rational_field::scalar_type;
/// Polynomial (with rational coefficients) scalar type
//using rational_poly_scalar = lal::rational_poly;

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
    constexpr explicit
    DeviceSpecification(DeviceCategory cat, uint32_t vendor_id)
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
    APRationalPolynomial = Polynomial// | ArbitraryPrecisionRational

};

/**
 * @brief Basic information for identifying the type, size, alignment, and
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
    uint8_t alignment;
    uint8_t lanes = 1;
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

void ROUGHPY_PLATFORM_EXPORT intrusive_ptr_add_ref(const DeviceHandle* device
) noexcept;
void ROUGHPY_PLATFORM_EXPORT intrusive_ptr_release(const DeviceHandle* device
) noexcept;

class BufferInterface;
class Buffer;
class EventInterface;
class Event;
class HostDeviceHandle;
class KernelArgument;
class KernelInterface;
class Kernel;
class MemoryView;
class QueueInterface;
class Queue;

using Device = Rc<const DeviceHandle>;
using HostDevice = Rc<const HostDeviceHandle>;

ROUGHPY_PLATFORM_EXPORT HostDevice get_host_device();
ROUGHPY_PLATFORM_EXPORT Device get_default_device();
ROUGHPY_PLATFORM_EXPORT optional<Device>
get_device(const DeviceSpecification& spec);

constexpr bool operator==(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}
constexpr bool operator!=(const DeviceInfo& lhs, const DeviceInfo& rhs) noexcept
{
    return lhs.device_type != rhs.device_type || lhs.device_id != rhs.device_id;
}

constexpr bool operator==(const TypeInfo& lhs, const TypeInfo& rhs) noexcept
{
    return lhs.code == rhs.code && lhs.bytes == rhs.bytes
            && lhs.lanes == rhs.lanes;
}

constexpr bool operator!=(const TypeInfo& lhs, const TypeInfo& rhs) noexcept
{
    return lhs.code != rhs.code || lhs.bytes != rhs.bytes
            || lhs.lanes != rhs.lanes;
}




namespace dtl {
template <typename I>
struct integral_code_impl {
    static constexpr TypeCode value
            = is_signed_v<I> ? TypeCode::Int : TypeCode::UInt;
};

struct floating_code_impl {
    static constexpr TypeCode value = TypeCode::Float;
};

struct unknown_code_impl {
};

template <typename T>
struct not_integral_code_impl : public conditional_t<
                                        is_floating_point_v<T>,
                                        floating_code_impl,
                                        unknown_code_impl> {
};

template <typename T>
struct type_code_of_impl : public conditional_t<
                                   is_integral_v<T>,
                                   integral_code_impl<T>,
                                   not_integral_code_impl<T>> {
};


template <>
struct type_code_of_impl<half>
{
    static constexpr TypeCode value = TypeCode::Float;
};

template <>
struct type_code_of_impl<bfloat16>
{
    static constexpr TypeCode value = TypeCode::BFloat;
};

template <>
struct type_code_of_impl<rational_scalar_type>
{
    static constexpr TypeCode value = TypeCode::ArbitraryPrecisionRational;
};

template <>
struct type_code_of_impl<rational_poly_scalar>
{
    static constexpr TypeCode value = TypeCode::APRationalPolynomial;
};


}// namespace dtl

template <typename T>
constexpr TypeCode type_code_of() noexcept
{
    return dtl::type_code_of_impl<T>::value;
}

/**
 * @brief Get the type info struct relating to the given type
 * @tparam T Type to query
 * @return TypeInfo struct containing information about the type.
 */
template <typename T>
constexpr TypeInfo type_info() noexcept
{
    return {type_code_of<T>(),
            static_cast<uint8_t>(sizeof(T)),
            static_cast<uint8_t>(alignof(T)),
            1U};
}

ROUGHPY_PLATFORM_EXPORT
std::ostream& operator<<(std::ostream& os, const TypeInfo& code);


RPY_SERIAL_SERIALIZE_FN_EXT(TypeInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("code", value.code);
    RPY_SERIAL_SERIALIZE_NVP("bytes", value.bytes);
    RPY_SERIAL_SERIALIZE_NVP("alignment", value.alignment);
    RPY_SERIAL_SERIALIZE_NVP("lanes", value.lanes);
}


}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
