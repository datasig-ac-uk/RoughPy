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

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/core/errors.h>
#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/slice.h>

#include <boost/uuid/uuid.hpp>

#if RPY_HAS_INCLUDE("roughpy_platform_export.h")
#  include "roughpy_platform_export.h"
#elif RPY_HAS_INCLUDE(<roughpy / platform / roughpy_platform_export.h>)
#  include <roughpy/platform/roughpy_platform_export.h>
#endif

#define ROUGHPY_DEVICES_EXPORT ROUGHPY_PLATFORM_EXPORT
#define ROUGHPY_DEVICES_NO_EXPORT ROUGHPY_PLATFORM_NO_EXPORT
#define ROUGHPY_DEVICES_DEPRECATED ROUGHPY_PLATFORM_DEPRECATED
#define ROUGHPY_DEVICES_DEPRECATED_EXPORT ROUGHPY_PLATFORM_DEPRECATED_EXPORT

#ifdef ROUGHPY_PLATFORM_NO_DEPRECATED
#define ROUGHPY_DEVICES_NO_DEPRECATED
#endif

namespace rpy {
namespace devices {

using dindex_t = int;
using dsize_t = unsigned int;

/**
 * @brief Enumeration of different device categories
 *
 * This enumeration represents the different categories of devices that scalar
 * data may be allocated on. Each category has an associated integer value
 * assigned to it.
 *
 * The available categories are:
 * - CPU: Represents a Central Processing Unit device.
 * - GPU: Represents a Graphics Processing Unit device.
 * - FPGA: Represents a Field-Programmable Gate Array device.
 * - DSP: Represents a Digital Signal Processor device.
 * - AIP: Represents an Application-Specific Integrated Circuit device.
 * - Other: Represents an unspecified category of devices.
 *
 * The integer values assigned to each category are implementation specific and
 * may vary across platforms. It is recommended to use the enumeration values
 * defined in this enum class instead of their integer representation.
 */
enum class DeviceCategory : int32_t
{
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    DSP = 3,
    AIP = 4,
    Other = 5
};

/**
 * @brief Enumeration of different types of device identification
 *
 * This enumeration represents the different types of identification that can be
 * used to identify a device. Each type has an associated integer value assigned
 * to it.
 *
 * The available types are:
 * - None: Represents no specific device identification.
 * - VendorID: Represents the identification of a device through its vendor ID.
 * - UUID: Represents the identification of a device through its universally
 * unique identifier.
 * - PCI: Represents the identification of a device through its PCI information.
 *
 * The integer values assigned to each type are implementation specific and
 * may vary across platforms. It is recommended to use the enumeration values
 * defined in this enum class instead of their integer representation.
 */
enum class DeviceIdType : int32_t
{
    None = 0,
    VendorID = 1,
    UUID = 2,
    PCI = 3
};

/**
 * @brief Structure representing PCI bus information
 *
 * This structure contains information about a PCI bus. It includes the PCI
 * domain, PCI bus number, PCI device number, and PCI function number.
 */
struct PCIBusInfo {
    uint32_t pci_domain;
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
};

/**
 * @brief Class representing a device specification.
 *
 * This class encapsulates information about a device specification, including
 * its category, identification type, identification value, and additional
 * details like vendor ID, UUID, and PCI bus information.
 *
 * The device specification can be used to query and filter devices based on
 * their specifications.
 */
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
    APRationalPolynomial = Polynomial,// | ArbitraryPrecisionRational

    KeyType = 32,
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

class BufferInterface;

class Buffer;

class EventInterface;

class Event;

class HostDeviceHandle;

class KernalArgument;

class KernelInterface;

class Kernel;

class QueueInterface;

class Queue;

class AlgorithmDrivers;

using Device = Rc<const DeviceHandle>;
using HostDevice = Rc<const HostDeviceHandle>;
using AlgorithmDriversPtr = Rc<const AlgorithmDrivers>;

ROUGHPY_DEVICES_EXPORT HostDevice get_host_device();

ROUGHPY_DEVICES_EXPORT Device get_default_device();

ROUGHPY_DEVICES_EXPORT optional<Device>
get_device(const DeviceSpecification& spec);

ROUGHPY_DEVICES_EXPORT
RPY_NO_DISCARD Device get_best_device(Slice<Device> devices);

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

template <typename T>
struct type_size_of_impl {
    static constexpr dimn_t value = sizeof(T);
};

template <typename T>
struct type_align_of_impl {
    static constexpr dimn_t value = alignof(T);
};

template <typename T>
struct type_lanes_of_impl {
    static constexpr dimn_t value = 1;
};

}// namespace dtl

/**
 * @brief Retrieves the type code of the template parameter type.
 *
 * This function determines the type code of the template parameter type `T`,
 * based on its characteristics. The type code represents the category of the
 * type and is defined in the `TypeCode` enum class.
 *
 * @tparam T The type for which to retrieve the type code.
 * @return The type code of the template parameter type `T`.
 *
 * @see TypeCode
 */
template <typename T>
constexpr TypeCode type_code_of() noexcept
{
    return dtl::type_code_of_impl<T>::value;
}

/**
 * @brief Get the size of the type T in bytes.
 *
 * This method returns the size of the type T in bytes. It uses the
 * type_size_of_impl template class to calculate the size of the type.
 *
 * @tparam T The type whose size needs to be calculated.
 * @return The size of the type T in bytes.
 */
template <typename T, typename I = dimn_t>
constexpr I type_size_of() noexcept
{
    return static_cast<I>(dtl::type_size_of_impl<T>::value);
}

/**
 * @brief Returns the alignment of the given type.
 *
 * This function determines the alignment of the type T. The alignment is
 * defined as the number of bytes that are guaranteed to be aligned with the
 * start of each object of that type. The alignment is the maximum of the
 * alignments of all the types in the type hierarchy of T. The alignment of a
 * scalar type is implementation-defined, but at least 1.
 *
 * @tparam T The type for which to determine the alignment.
 * @return The alignment of the type T.
 *
 * @see TypeInfo
 * @see type_info()
 */
template <typename T, typename I = dimn_t>
constexpr I type_align_of() noexcept
{
    return static_cast<I>(dtl::type_align_of_impl<T>::value);
}

template <typename T, typename I = dimn_t>
/**
 * @brief Returns the number of lanes for the given type.
 *
 * This method returns the number of lanes for the given type. The number of
 * lanes represents the number of parallel elements that can be processed
 * simultaneously in SIMD (Single Instruction, Multiple Data) operations.
 *
 * SIMD operations can be used to perform vectorized computations, where
 * multiple values are processed in parallel using the same operation. The
 * number of lanes determines the size of the vectors that can be processed in a
 * single SIMD operation.
 *
 * The return type of this method is a template parameter `I`, which represents
 * an integer type. The return value is obtained by casting the number of lanes
 * to the specified integer type. The implicit conversion from `dimn_t` (a type
 * alias for an unsigned integer) to `I` is performed using `static_cast`.
 *
 * This method is marked as `constexpr` and `noexcept`, indicating that it can
 * be evaluated at compile-time and that it does not throw any exceptions.
 *
 * @return The number of lanes for the given type.
 *
 * @see type_code_of, type_size_of, type_align_of
 */
constexpr I type_lanes_of() noexcept
{
    return static_cast<I>(dtl::type_lanes_of_impl<T>::value);
}

/**
 * @brief Retrieves the TypeInfo object for the specified type T.
 *
 * This method returns the TypeInfo object, which contains information about the
 * type T such as its code, size, alignment, and number of lanes. The type code,
 * size, and alignment are determined using the type_code_of, type_size_of, and
 * type_align_of functions respectively. The number of lanes is determined using
 * the type_lanes_of function.
 *
 * @tparam T The type for which the TypeInfo is retrieved.
 * @return The TypeInfo object for the specified type T.
 *
 * @note The type T must be a complete type.
 * @note The TypeInfo object returned is implementation-specific and may vary
 * across platforms.
 */
template <typename T>
constexpr TypeInfo type_info() noexcept
{
    return {type_code_of<T>(),
            type_size_of<T, uint8_t>(),
            type_align_of<T, uint8_t>(),
            type_lanes_of<T, uint8_t>()};
}

ROUGHPY_DEVICES_EXPORT
std::ostream& operator<<(std::ostream& os, const TypeInfo& code);

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
