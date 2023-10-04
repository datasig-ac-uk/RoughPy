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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_FWD_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_FWD_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <memory>

namespace rpy {
namespace algebra {

enum class ImplementationType
{
    Owned,
    Borrowed,
    DeviceOwned,
    DeviceBorrowed
};

enum class VectorType : uint16_t
{
    Dense,
    Sparse
};

template <VectorType>
struct vector_type_tag {};

/**
 * @brief Different algebra types required by RoughPy
 */
enum class AlgebraType : uint16_t
{
    FreeTensor,
    ShuffleTensor,
    Lie,
    FreeTensorBundle,
    ShuffleTensorBundle,
    LieBundle
};

template <AlgebraType>
struct algebra_type_tag {};

template <typename Interface>
struct algebra_access;

template <typename Wrapper, typename Algebra>
struct algebra_info;

template <typename Algebra>
class AlgebraIterator;

template <typename PrimaryInterface>
class Basis;

namespace dtl {

class AlgebraInterfaceBase;

}
using RawUnspecifiedAlgebraType = dtl::AlgebraInterfaceBase*;
using ConstRawUnspecifiedAlgebraType = const dtl::AlgebraInterfaceBase*;
using UnspecifiedAlgebraType = std::unique_ptr<dtl::AlgebraInterfaceBase>;

class FreeTensor;

class Lie;

class ShuffleTensor;

class FreeTensorBundle;

class LieBundle;

class ShuffleTensorBundle;


RPY_SERIAL_SAVE_FN_EXT(VectorType) {
    RPY_SERIAL_SERIALIZE_BARE(static_cast<uint16_t>(value));
}

RPY_SERIAL_LOAD_FN_EXT(VectorType) {
    uint16_t tmp;
    RPY_SERIAL_SERIALIZE_BARE(tmp);
    value = static_cast<VectorType>(tmp);
}


RPY_SERIAL_SAVE_FN_EXT(AlgebraType) {
    RPY_SERIAL_SERIALIZE_BARE(static_cast<uint16_t>(value));
}

RPY_SERIAL_LOAD_FN_EXT(AlgebraType) {
    uint16_t tmp;
    RPY_SERIAL_SERIALIZE_BARE(tmp);
    value = static_cast<AlgebraType>(tmp);
}

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::VectorType,
                            rpy::serial::specialization::non_member_load_save);

RPY_SERIAL_SPECIALIZE_TYPES(rpy::algebra::AlgebraType,
                            rpy::serial::specialization::non_member_load_save);

#endif// ROUGHPY_ALGEBRA_ALGEBRA_FWD_H_
