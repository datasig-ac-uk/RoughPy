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
// Created by user on 06/03/23.
//

#include "hall_set_size.h"

using namespace rpy;
using namespace rpy::algebra;

HallSetSizeHelper::HallSetSizeHelper(deg_t width, deg_t depth)
    : m_width(width), m_depth(depth)
{
    if (static_cast<dimn_t>(m_depth) > m_mobius.size()) {
        std::vector<bool> tmp;
        tmp.resize(m_depth / 2, true);
        tmp[0] = false;

        auto bound = m_depth / 2;
        for (int i = 2; i < bound; ++i) {
            for (int j = 2, m = 2 * i; m < bound; ++j, m = i * j) {
                tmp[m] = false;
            }
        }
        std::vector<int> primes;
        primes.reserve(bound);
        for (int i = 2; i < bound; ++i) {
            if (tmp[i]) { primes.push_back(i); }
        }

        /*
         * The algorithm for computing the Mobius function is taken from
         *
         * Lioen, W. M., & van de Lune, J. (1994). Systematic computations on
         * Mertens’ conjecture and Dirichlet’s divisor problem by vectorized
         * sieving. From universal morphisms to megabytes: a Baayen space
         * Odyssey, 421-432.
         */

        auto old_size = static_cast<int>(m_mobius.size());
        m_mobius.resize(m_depth + 1, 1);
        for (auto& prime : primes) {
            for (deg_t i = (old_size / prime) + 1, m = i * prime; m <= m_depth;
                 ++i) {
                m_mobius[m] *= -prime;
            }
            auto p2 = prime * prime;
            for (deg_t i = (old_size / p2) + 1, m = i * p2; m <= m_depth; ++i) {
                m_mobius[m] = 0;
            }
        }
        for (deg_t i = old_size + 1; i <= m_depth; ++i) {
            if (abs(m_mobius[i]) != i) { m_mobius[i] = -m_mobius[i]; }
            // Mobius[i] = sign(Mobius[i])
            m_mobius[i] = static_cast<int>(0 < m_mobius[i])
                    - static_cast<int>(m_mobius[i] < 0);
        }
    }
}

/*
 * This is a special version of power for use in the computation of
 * the Lie algebra degree size. In that case, we cannot have an exponent of 0
 * which eliminates one branch from the function.
 */
static inline dimn_t power(deg_t base, int exp)
{
    if (exp == 1) { return base; }
    return ((exp % 2 == 1) ? base : 1) * power(base, exp / 2)
            * power(base, exp / 2);
}

dimn_t HallSetSizeHelper::operator()(int k)
{
    dimn_t result = 0;

    for (deg_t i = 1; i <= k; ++i) {
        auto divs = div(k, i);
        if (divs.rem == 0) {
            auto comp = k / divs.quot;
            result += static_cast<dimn_t>(
                    m_mobius[divs.quot] * power(m_width, comp)
            );
        }
    }

    RPY_DBG_ASSERT(result % k == 0);
    return result / k;
}
