// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 26/04/23.
//

#ifndef ROUGHPY_SCALARS_SRC_HALF_RANDOM_GENERATOR_H
#define ROUGHPY_SCALARS_SRC_HALF_RANDOM_GENERATOR_H

#include "random_impl.h"
#include <roughpy/scalars/random.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include "standard_random_generator.h"

#include <mutex>
#include <random>
#include <sstream>
#include <vector>

#include <pcg_random.hpp>

namespace rpy {
namespace scalars {

template <typename BitGenerator>
class StandardRandomGenerator<half, BitGenerator> : public RandomGenerator
{
    using scalar_type = half;
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
    void set_state(string_view state) override;
    std::vector<uint64_t> get_seed() const override;
    string get_type() const override;
    string get_state() const override;

    OwnedScalarArray uniform_random_scalar(ScalarArray lower, ScalarArray upper,
                                           dimn_t count) const override;
    OwnedScalarArray normal_random(Scalar loc, Scalar scale,
                                   dimn_t count) const override;
};

template <typename BitGenerator>
void StandardRandomGenerator<half, BitGenerator>::set_state(string_view state)
{
    std::stringstream ss(string{state});
    ss >> m_generator;
}
template <typename BitGenerator>
string StandardRandomGenerator<half, BitGenerator>::get_state() const
{
    std::stringstream ss;
    ss << m_generator;
    return ss.str();
}

template <typename BitGenerator>
StandardRandomGenerator<half, BitGenerator>::StandardRandomGenerator(
        const ScalarType* stype, Slice<uint64_t> seed)
    : RandomGenerator(stype), m_seed{seed[0]},
      m_generator(BitGenerator(seed[0]))
{
    RPY_CHECK(p_type = ScalarType::of<half>());
    RPY_CHECK(seed.size() >= 1);
}

template <typename BitGenerator>
void StandardRandomGenerator<half, BitGenerator>::set_seed(
        Slice<uint64_t> seed_data)
{
    RPY_CHECK(seed_data.size() >= 1);
    m_generator.seed(seed_data[0]);
    m_seed = {seed_data[0]};
}
template <typename BitGenerator>
std::vector<uint64_t>
StandardRandomGenerator<half, BitGenerator>::get_seed() const
{
    return {m_seed[0]};
}
template <typename BitGenerator>
std::string StandardRandomGenerator<half, BitGenerator>::get_type() const
{
    return std::string(dtl::rng_type_getter<BitGenerator>::name);
}

template <typename BitGenerator>
OwnedScalarArray
StandardRandomGenerator<half, BitGenerator>::uniform_random_scalar(
        ScalarArray lower, ScalarArray upper, dimn_t count) const
{
    std::vector<std::uniform_real_distribution<float>> dists;

    RPY_CHECK(lower.size() == upper.size());

    dists.reserve(lower.size());
    for (dimn_t i = 0; i < lower.size(); ++i) {
        dists.emplace_back(static_cast<float>(scalar_cast<half>(lower[i])),
                           static_cast<float>(scalar_cast<half>(upper[i])));
    }

    OwnedScalarArray result(p_type, count * dists.size());

    auto* out = result.raw_cast<half*>();
    for (dimn_t i = 0; i < count; ++i) {
        for (auto& dist : dists) { ::new (out++) half(dist(m_generator)); }
    }

    return result;
}
template <typename BitGenerator>
OwnedScalarArray StandardRandomGenerator<half, BitGenerator>::normal_random(
        Scalar loc, Scalar scale, dimn_t count) const
{

    OwnedScalarArray result(p_type, count);
    std::normal_distribution<float> dist(scalar_cast<half>(loc),
                                         scalar_cast<half>(scale));

    auto* out = result.raw_cast<half*>();
    for (dimn_t i = 0; i < count; ++i) {
        ::new (out++) half(dist(m_generator));
    }

    return result;
}

extern template class StandardRandomGenerator<half, std::mt19937_64>;

extern template class StandardRandomGenerator<half, pcg64>;

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_HALF_RANDOM_GENERATOR_H
