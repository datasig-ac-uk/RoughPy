//
// Created by sam on 11/17/23.
//

#ifndef BFLOAT16_RANDOM_GENERATOR_H
#define BFLOAT16_RANDOM_GENERATOR_H

#include "random/standard_random_generator.h"

#include "roughpy/core/check.h"                    // for throw_exception

#include "scalar_types.h"


namespace rpy {
namespace scalars {

template <typename BitGenerator>
class StandardRandomGenerator<bfloat16, BitGenerator> : public RandomGenerator
{
    using scalar_type = bfloat16;
    using bit_generator = BitGenerator;

    std::vector<seed_int_t> m_seed;

    mutable BitGenerator m_generator;
    mutable std::mutex m_lock;

public:
    StandardRandomGenerator(const ScalarType* stype, Slice<seed_int_t> seed);

    void set_seed(Slice<seed_int_t> seed_data) override;
    void set_state(string_view state) override;
    RPY_NO_DISCARD std::vector<seed_int_t> get_seed() const override;
    RPY_NO_DISCARD std::string get_type() const override;
    RPY_NO_DISCARD std::string get_state() const override;
    RPY_NO_DISCARD ScalarArray uniform_random_scalar(const ScalarArray& lower,
        const ScalarArray& upper,
        dimn_t count) const override;
    RPY_NO_DISCARD ScalarArray normal_random(Scalar loc,
                                             Scalar scale,
                                             dimn_t count) const override;
};

template <typename BitGenerator>
StandardRandomGenerator<bfloat16, BitGenerator>::StandardRandomGenerator(
    const ScalarType* stype,
    Slice<seed_int_t> seed)
    : RandomGenerator(stype)
{
    RPY_CHECK(p_type == *ScalarType::of<bfloat16>());
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
    } else { m_seed = seed; }

    m_generator = BitGenerator(m_seed[0]);
}

template <typename BitGenerator>
void StandardRandomGenerator<bfloat16, BitGenerator>::set_seed(
    Slice<seed_int_t> seed_data)
{
    RPY_CHECK(seed_data.size() >= 1);

    m_generator.seed(seed_data[0]);
    m_seed = {seed_data[0]};
}

template <typename BitGenerator>
void StandardRandomGenerator<bfloat16, BitGenerator>::
set_state(string_view state)
{
    // add a linebreak char to terminate the string to avoid a bug in the pcg
    // stream read function
    std::stringstream ss(string{state} + '\n');
    ss >> m_generator;
}

template <typename BitGenerator>
std::vector<seed_int_t> StandardRandomGenerator<bfloat16, BitGenerator>::
get_seed() const { return {m_seed[0]}; }

template <typename BitGenerator>
std::string StandardRandomGenerator<bfloat16, BitGenerator>::get_type() const
{
    return string(dtl::rng_type_getter<BitGenerator>::name);
}

template <typename BitGenerator>
std::string StandardRandomGenerator<bfloat16, BitGenerator>::get_state() const
{
    std::stringstream ss;
    ss << m_generator;
    return ss.str();
}

template <typename BitGenerator>
ScalarArray StandardRandomGenerator<bfloat16, BitGenerator>::uniform_random_scalar(
    const ScalarArray& lower,
    const ScalarArray& upper,
    dimn_t count) const
{
    std::vector<std::uniform_real_distribution<float>> dists;

    if (lower.size() != upper.size()) {
        RPY_THROW(
            std::invalid_argument,
            "mismatch dimensions between lower and upper limits"
        );
    }

    dists.reserve(lower.size());
    for (dimn_t i = 0; i < lower.size(); ++i) {
        dists.emplace_back(
            scalar_cast<float>(lower[i]),
            scalar_cast<float>(upper[i])
        );
    }

    ScalarArray result(p_type, count * dists.size());

    auto out_slice = result.as_mut_slice<scalar_type>();
    auto* out = out_slice.data();
    for (dimn_t i = 0; i < count; ++i) {
        for (auto& dist : dists) {
            construct_inplace(out++, scalar_type(dist(m_generator)));
        }
    }

    return result;
}

template <typename BitGenerator>
ScalarArray StandardRandomGenerator<bfloat16, BitGenerator>::normal_random(
    Scalar loc,
    Scalar scale,
    dimn_t count) const
{
    ScalarArray result(p_type, count);
    std::normal_distribution<float> dist(
        scalar_cast<float>(loc),
        scalar_cast<float>(scale)
    );

    auto out_slice = result.as_mut_slice<scalar_type>();
    auto* out = out_slice.data();
    for (dimn_t i = 0; i < count; ++i) {
        construct_inplace(out++, scalar_type(dist(m_generator)));
    }

    return result;
}

extern template class StandardRandomGenerator<bfloat16, std::mt19937_64>;
extern template class StandardRandomGenerator<bfloat16, pcg64>;


}// scalars
}// rpy

#endif //BFLOAT16_RANDOM_GENERATOR_H
