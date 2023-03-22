//
// Created by user on 25/02/23.
//

#ifndef ROUGHPY_SCALARS_SCALARS_PREDEF_H
#define ROUGHPY_SCALARS_SCALARS_PREDEF_H

#include "roughpy_scalars_export.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <roughpy/config/traits.h>

namespace rpy {
namespace scalars {


using scalar_t = double;
using dimn_t = std::size_t;
using idimn_t = std::ptrdiff_t;
using key_type = std::size_t;


struct signed_size_type_marker {};
struct unsigned_size_type_marker {};

/**
 * @brief Code for different device types
 *
 * These codes are chosen to be compatible with the DLPack
 * array interchange protocol. They enumerate the various different
 * device types that scalar data may be allocated on. This code goes
 * with a 32bit integer device ID, which is implementation specific.
 */
enum class ScalarDeviceType : std::int32_t {
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
 * @brief Type codes for different scalar types.
 *
 * These are chosen to be compatible with the DLPack
 * array interchange protocol. Rational types will
 * be encoded as OpaqueHandle, since they're not simple
 * data. Some of these types might not be compatible with
 * this library.
 */
enum class ScalarTypeCode : std::uint8_t {
    Int = 0U,
    UInt = 1U,
    Float = 2U,
    OpaqueHandle = 3U,
    BFloat = 4U,
    Complex = 5U,
    Bool = 6U
};

/**
 * @brief Device type/id pair to identify a device
 *
 *
 */
struct ScalarDeviceInfo {
    ScalarDeviceType device_type;
    std::int32_t device_id;
};

/**
 * @brief Basic information for identifying the type, size, and
 * configuration of a scalar.
 *
 * Based on, and compatible with, the DlDataType struct from the
 * DLPack array interchange protocol.
 */
struct BasicScalarInfo {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};


/**
 * @brief A collection of basic information for identifying a scalar type.
 */
struct ScalarTypeInfo {
    BasicScalarInfo basic_info;
    ScalarDeviceInfo device;
    std::string name;
    std::string id;
    std::size_t n_bytes;
    std::size_t alignment;
};



// Forward declarations

class ScalarType;
class ScalarInterface;

class ScalarPointer;
class Scalar;
class ScalarArray;
class OwnedScalarArray;
class KeyScalarArray;
class ScalarStream;

class RandomGenerator;

template <typename T>
inline traits::remove_cv_ref_t<T> scalar_cast(const Scalar& arg);

} // namespace scalars
} // namespace rpy


#endif //ROUGHPY_SCALARS_SCALARS_PREDEF_H
