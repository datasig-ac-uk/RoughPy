
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

#include <roughpy/platform/archives.h>
#include <roughpy/scalars/scalar_serialization.h>
#include <roughpy/scalars/scalars_fwd.h>
#include <roughpy/scalars/scalar_types.h>

using namespace rpy;
using namespace rpy::scalars;

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::save(
        cereal::JSONOutputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
) const
{
    RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
    RPY_SERIAL_SERIALIZE_SIZE(nbytes());
    archive.saveBinaryValue(limbs(), nbytes(), "data");
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::load(
        cereal::JSONInputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
)
{
    bool is_negative;
    dimn_t size;

    RPY_SERIAL_SERIALIZE_VAL(is_negative);
    {
        // This is probably redundant, but keep the type system happy.
        serial::size_type tmp_size;
        RPY_SERIAL_SERIALIZE_SIZE(tmp_size);
        size = static_cast<dimn_t>(tmp_size);
    }

    if (size > 0) {
        auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
        archive.loadBinaryValue(resize(n_limbs), size, "data");
        finalize(n_limbs, is_negative);
    }
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::save(
        cereal::XMLOutputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
) const
{
    RPY_SERIAL_SERIALIZE_NVP("is_negative", is_negative());
    RPY_SERIAL_SERIALIZE_SIZE(nbytes());
    archive.saveBinaryValue(limbs(), nbytes(), "data");
}

template <typename Integer>
void rpy::scalars::dtl::MPIntegerSerializationHelper<Integer>::load(
        cereal::XMLInputArchive& archive,
        const std::uint32_t RPY_UNUSED_VAR version
)
{
    bool is_negative;
    dimn_t size;

    RPY_SERIAL_SERIALIZE_VAL(is_negative);

    {
        // This is probably redundant, but keep the type system happy.
        serial::size_type tmp_size;
        RPY_SERIAL_SERIALIZE_SIZE(tmp_size);
        size = static_cast<dimn_t>(tmp_size);
    }
    if (size > 0) {
        auto n_limbs = (size + sizeof(limbs_t) - 1) / sizeof(limbs_t);
        archive.loadBinaryValue(resize(n_limbs), size, "data");
        finalize(n_limbs, is_negative);
    }
}

#define RPY_EXPORT_MACRO ROUGHPY_SCALARS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::half
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::bfloat16
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::indeterminate_type
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::monomial
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::rational_poly_scalar
#define RPY_SERIAL_EXTERNAL cereal
#define RPY_SERIAL_DO_SPLIT
#define RPY_SERIAL_NO_VERSION
#include <roughpy/platform/serialization_instantiations.inl>
