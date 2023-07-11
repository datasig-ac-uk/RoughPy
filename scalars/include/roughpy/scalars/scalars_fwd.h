// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

//
// Created by user on 25/02/23.
//

#ifndef ROUGHPY_SCALARS_SCALARS_PREDEF_H
#define ROUGHPY_SCALARS_SCALARS_PREDEF_H

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include <Eigen/Core>
#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/packed_integer.h>
#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/polynomial_basis.h>

#ifdef LAL_NO_USE_GMP
#  define RPY_USING_GMP 0
#else
#  define RPY_USING_GMP 1
#endif

namespace rpy {
namespace scalars {

/// IEEE half-precision floating point type
using Eigen::half;

/// BFloat16 (truncated) floating point type
using Eigen::bfloat16;

/// Rational scalar type
using rational_scalar_type = lal::rational_field::scalar_type;

/// Monomial key-type of polynomials
using monomial = lal::monomial;

/// Indeterminate type for monomials
using indeterminate_type = typename monomial::letter_type;

/// Polynomial (with rational coefficients) scalar type
using rational_poly_scalar = lal::polynomial<lal::rational_field>;

/// Marker for signed size type (ptrdiff_t)
struct signed_size_type_marker {
};

/// Marker for unsigned size type (size_t)
struct unsigned_size_type_marker {
};

/**
 * @brief Code for different device types
 *
 * These codes are chosen to be compatible with the DLPack
 * array interchange protocol. They enumerate the various different
 * device types that scalar data may be allocated on. This code goes
 * with a 32bit integer device ID, which is implementation specific.
 */
enum class ScalarDeviceType : int32_t
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
 * @brief Type codes for different scalar types.
 *
 * These are chosen to be compatible with the DLPack
 * array interchange protocol. Rational types will
 * be encoded as OpaqueHandle, since they're not simple
 * data. Some of these types might not be compatible with
 * this library.
 */
enum class ScalarTypeCode : uint8_t
{
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
 * DLPack array interchange protocol. The lanes parameter will
 * usually be set to 1, and is not generally used by RoughPy.
 */
struct BasicScalarInfo {
    ScalarTypeCode code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

/**
 * @brief A collection of basic information for identifying a scalar type.
 */
struct ScalarTypeInfo {
    string name;
    string id;
    std::size_t n_bytes;
    std::size_t alignment;
    BasicScalarInfo basic_info;
    ScalarDeviceInfo device;
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

class BlasInterface;

template <typename T>
inline remove_cv_ref_t<T> scalar_cast(const Scalar& arg);

using conversion_function
        = std::function<void(ScalarPointer, ScalarPointer, dimn_t)>;

constexpr bool
operator==(const ScalarDeviceInfo& lhs, const ScalarDeviceInfo& rhs) noexcept
{
    return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

constexpr bool
operator==(const BasicScalarInfo& lhs, const BasicScalarInfo& rhs) noexcept
{
    return lhs.code == rhs.code && lhs.bits == rhs.bits
            && lhs.lanes == rhs.lanes;
}

/**
 * @brief Register a new type with the scalar type system
 * @param type Pointer to newly created ScalarType
 *
 *
 */
RPY_EXPORT
void register_type(const ScalarType* type);

/**
 * @brief Get a type registered with the scalar type system
 * @param id Id string of type to be retrieved
 * @return pointer to ScalarType representing id
 */
RPY_EXPORT
const ScalarType* get_type(const string& id);

RPY_EXPORT
const ScalarType* get_type(const string& id, const ScalarDeviceInfo& device);

/**
 * @brief Get a list of all registered ScalarTypes
 * @return vector of ScalarType pointers.
 */
RPY_NO_DISCARD RPY_EXPORT std::vector<const ScalarType*> list_types();

RPY_NO_DISCARD RPY_EXPORT const ScalarTypeInfo& get_scalar_info(string_view id);

RPY_NO_DISCARD RPY_EXPORT const std::string&
id_from_basic_info(const BasicScalarInfo& info);

RPY_NO_DISCARD RPY_EXPORT const conversion_function&
get_conversion(const string& src_id, const string& dst_id);

RPY_EXPORT
void register_conversion(
        const string& src_id, const string& dst_id,
        conversion_function converter
);

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALARS_PREDEF_H
