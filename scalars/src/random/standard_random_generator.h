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
// Created by user on 20/03/23.
//

#ifndef ROUGHPY_SCALARS_SRC_STANDARD_RANDOM_GENERATOR_H
#define ROUGHPY_SCALARS_SRC_STANDARD_RANDOM_GENERATOR_H

#include <roughpy/scalars/random.h>

#include <mutex>
#include <random>
#include <sstream>
#include <vector>

#include <roughpy/core/alloc.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_types.h>

#include "random_impl.h"
#include <pcg_random.hpp>
#include <pcg_uint128.hpp>

namespace pcg_extras {

template <typename Uint, typename UintX2, typename I>
constexpr uint_x4<Uint, UintX2>
operator|(const uint_x4<Uint, UintX2>& lhs, const I& rhs)
{
    return lhs | static_cast<uint_x4<Uint, UintX2>>(rhs);
}

}// namespace pcg_extras

namespace rpy {
namespace scalars {

template <typename ScalarImpl, typename BitGenerator>
class StandardRandomGenerator : public RandomGenerator
{
    using scalar_type = ScalarImpl;
    using bit_generator = BitGenerator;

    std::vector<uint64_t> m_seed;

    mutable BitGenerator m_generator;
    mutable std::mutex m_lock;

public:
    StandardRandomGenerator(const ScalarType* stype, Slice<uint64_t> seed);

    StandardRandomGenerator(const StandardRandomGenerator&) = delete;
    StandardRandomGenerator(StandardRandomGenerator&&) noexcept = delete;
    StandardRandomGenerator& operator=(const StandardRandomGenerator&) = delete;
    StandardRandomGenerator& operator=(StandardRandomGenerator&&) noexcept
    = delete;

    void set_seed(Slice<uint64_t> seed_data) override;
    std::vector<uint64_t> get_seed() const override;
    string get_type() const override;
    string get_state() const override;

    ScalarArray uniform_random_scalar(
        const ScalarArray& lower,
        const ScalarArray& upper,
        dimn_t count
    ) const override;
    ScalarArray
    normal_random(Scalar loc, Scalar scale, dimn_t count) const override;
    void set_state(string_view state) override;
};

template <typename ScalarImpl, typename BitGenerator>
string StandardRandomGenerator<ScalarImpl, BitGenerator>::get_state() const
{
    std::stringstream ss;
    ss << m_generator;
    return ss.str();
}

template <typename ScalarImpl, typename BitGenerator>
string StandardRandomGenerator<ScalarImpl, BitGenerator>::get_type() const
{
    return string(dtl::rng_type_getter<BitGenerator>::name);
}

template <typename ScalarImpl, typename BitGenerator>
StandardRandomGenerator<ScalarImpl, BitGenerator>::StandardRandomGenerator(
    const ScalarType* stype,
    Slice<uint64_t> seed
)
    : RandomGenerator(stype)
{
    RPY_CHECK(p_type == *ScalarType::of<ScalarImpl>());
    if (seed.empty()) {
        m_seed.resize(1);
        auto& s = m_seed[0];
        std::random_device dev;
        auto continue_bits = static_cast<idimn_t>(sizeof(seed_int_t) *
            CHAR_BIT);

        constexpr auto so_rd_int = static_cast<idimn_t>(sizeof(typename
            std::random_device::result_type) * CHAR_BIT);
        while (continue_bits > 0) {
            s <<= so_rd_int;
            s |= static_cast<seed_int_t>(dev());
            continue_bits -= so_rd_int;
        }
    } else {
        m_seed = seed;
    }

    m_generator = BitGenerator(m_seed[0]);
}

template <typename ScalarImpl, typename BitGenerator>
void StandardRandomGenerator<ScalarImpl, BitGenerator>::set_seed(
    Slice<uint64_t> seed_data
)
{
    RPY_CHECK(seed_data.size() >= 1);

    m_generator.seed(seed_data[0]);
    m_seed = {seed_data[0]};
}

template <typename ScalarImpl, typename BitGenerator>
void StandardRandomGenerator<ScalarImpl, BitGenerator>::set_state(
    string_view state
)
{
    // add a linebreak char to terminate the string to avoid a bug in the pcg
    // stream read function
    std::stringstream ss(string{state} + '\n');
    ss >> m_generator;
}

template <typename ScalarImpl, typename BitGenerator>
std::vector<uint64_t>
StandardRandomGenerator<ScalarImpl, BitGenerator>::get_seed() const
{
    return {m_seed[0]};
}

template <typename ScalarImpl, typename BitGenerator>
ScalarArray
StandardRandomGenerator<ScalarImpl, BitGenerator>::uniform_random_scalar(
    const ScalarArray& lower,
    const ScalarArray& upper,
    dimn_t count
) const
{

    std::vector<std::uniform_real_distribution<scalar_type>> dists;

    if (lower.size() != upper.size()) {
        RPY_THROW(
            std::invalid_argument,
            "mismatch dimensions between lower and upper limits"
        );
    }

    dists.reserve(lower.size());
    for (dimn_t i = 0; i < lower.size(); ++i) {
        dists.emplace_back(
            scalar_cast<scalar_type>(lower[i]),
            scalar_cast<scalar_type>(upper[i])
        );
    }

    ScalarArray result(p_type, count * dists.size());

    auto out_slice = result.as_mut_slice<scalar_type>();
    auto* out = out_slice.data();
    for (dimn_t i = 0; i < count; ++i) {
        for (auto& dist : dists) {
            construct_inplace(out++, dist(m_generator));
        }
    }

    return result;
}

template <typename ScalarImpl, typename BitGenerator>
ScalarArray
StandardRandomGenerator<ScalarImpl, BitGenerator>::normal_random(
    Scalar loc,
    Scalar scale,
    dimn_t count
) const
{
    ScalarArray result(p_type, count);
    std::normal_distribution<ScalarImpl> dist(
        scalar_cast<scalar_type>(loc),
        scalar_cast<scalar_type>(scale)
    );

    auto out_slice = result.as_mut_slice<scalar_type>();
    auto* out = out_slice.data();
    for (dimn_t i = 0; i < count; ++i) {
        construct_inplace(out++, dist(m_generator));
    }

    return result;
}

extern template class StandardRandomGenerator<float, std::mt19937_64>;

extern template class StandardRandomGenerator<float, pcg64>;

extern template class StandardRandomGenerator<double, std::mt19937_64>;

extern template class StandardRandomGenerator<double, pcg64>;

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_STANDARD_RANDOM_GENERATOR_H
