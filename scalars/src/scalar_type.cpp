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

#include "scalar_type.h"

#include "random.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalars_fwd.h"
#include "builtin_scalars.h"
#include "builtin_scalar_types.h"
#include "scalar/casts.h"

#include <charconv>

using namespace rpy;
using namespace rpy::scalars;

const ScalarType* ScalarType::for_info(const devices::TypeInfo& info)
{
    switch (info.code) {
        case devices::TypeCode::Int:
        case devices::TypeCode::UInt:
            // if (info.bytes <= 3) {
            // return *scalar_type_of<float>();
            // } else {
            return &double_type;
        // }
        case devices::TypeCode::Float:
            switch (info.bytes) {
                case 2: return &half_type;
                case 4: return &float_type;
                case 8: return &double_type;
                default:
                    break;
            }
        break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat: return &bfloat16_type;
        case devices::TypeCode::Complex: break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecision:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionComplex:
        case devices::TypeCode::ArbitraryPrecisionRational:
            return &arbitrary_precision_rational_type;
        case devices::TypeCode::APRationalPolynomial:
            return &arbitrary_precision_rational_polynomial_type;
    }

    RPY_THROW(std::runtime_error, "unsupported data type");
}

ScalarType::ScalarType(
        std::string name,
        std::string id,
        rpy::dimn_t alignment,
        devices::Device device,
        devices::TypeInfo type_info,
        rpy::scalars::RingCharacteristics characteristics
)
    : m_name(std::move(name)),
      m_id(std::move(id)),
      m_alignment(alignment),
      m_device(std::move(device)),
      m_info(type_info),
      m_characteristics(characteristics)
{}

ScalarType::~ScalarType() = default;

ScalarArray ScalarType::allocate(dimn_t count) const
{
    return ScalarArray(this, m_device->alloc(m_info, count));
}

void* ScalarType::allocate_single() const
{
    RPY_THROW(
            std::runtime_error,
            "single scalar allocation is not available "
            "for " + m_name
    );
    return nullptr;
}

void ScalarType::free_single(void* ptr) const
{
    RPY_THROW(
            std::runtime_error,
            "single scalar allocation is not available for " + m_name
    );
}

void ScalarType::convert_copy(
        rpy::scalars::ScalarArray& dst,
        const rpy::scalars::ScalarArray& src
) const
{
    if (dst.size() < src.size()) {
        RPY_THROW(std::runtime_error, "insufficient size for copy");
    }
    if (dst.device() != m_device) {
        RPY_THROW(std::runtime_error, "unable to copy into device memory");
    }

    if (!dtl::scalar_convert_copy(
                dst.mut_buffer().ptr(),
                dst.type_info(),
                src.buffer().ptr(),
                src.type_info(),
                dst.size()
        )) {
        RPY_THROW(std::runtime_error, "convert copy failed");
    }
}

const ScalarType* ScalarType::for_id(string_view id)
{
    return *ScalarType::of<double>();
}





std::unique_ptr<RandomGenerator>
ScalarType::get_rng(const string& bit_generator, Slice<seed_int_t> seed) const
{
    if (m_rng_getters.empty()) {
        RPY_THROW(
                std::runtime_error,
                "no random number generators available for type " + m_name
        );
    }

    if (bit_generator.empty()) {
        return m_rng_getters.begin()->second(this, seed);
    }

    const auto getter = m_rng_getters.find(bit_generator);
    if (getter != m_rng_getters.end()) { return getter->second(this, seed); }

    RPY_THROW(
            std::runtime_error,
            "no matching random number generator " + bit_generator
                    + " for type " + m_name
    );
}

void ScalarType::assign(ScalarArray& dst, Scalar value) const {}

const ScalarType* ScalarType::with_device(const devices::Device& device) const
{
    return this;
}

const ScalarType* ScalarType::rational_type() const noexcept { return this; }
void ScalarType::register_rng_getter(string name, rng_getter getter)
{
    m_rng_getters[name] = getter;
}
