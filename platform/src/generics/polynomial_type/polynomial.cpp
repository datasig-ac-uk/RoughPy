//
// Created by sam on 26/11/24.
//

#include "polynomial.h"

#include <algorithm>
#include <numeric>

#include "roughpy/core/hash.h"

#include "generics/multiprecision_types/mpq_string_rep.h"
#include "generics/multiprecision_types/mpz_hash.h"

using namespace rpy;
using namespace rpy::generics;

deg_t Polynomial::degree() const noexcept
{
    return std::accumulate(begin(), end(), 0, [](auto acc, const auto& pair) {
        return std::max(acc, pair.first.degree());
    });
}

void generics::poly_add_inplace(Polynomial& lhs, const Polynomial& rhs)
{
    const auto lend = lhs.end();
    dtl::RationalCoeff zero;
    for (const auto& [rhs_key, rhs_coeff] : rhs) {
        auto it = lhs.find(rhs_key);
        if (it != lend) {
            // Update existing term
            mpq_add(it->second.content, it->second.content, rhs_coeff.content);
            // If the resulting coefficient is zero, remove the term
            if (mpq_equal(it->second.content, zero.content)) { lhs.erase(it); }
        } else {
            // Insert new term
            lhs.emplace(rhs_key, rhs_coeff);
        }
    }
}

void generics::poly_sub_inplace(Polynomial& lhs, const Polynomial& rhs)
{
    const auto lend = lhs.end();
    dtl::RationalCoeff zero;
    for (const auto& [rhs_key, rhs_coeff] : rhs) {
        auto it = lhs.find(rhs_key);
        if (it != lend) {
            // Update existing term
            mpq_sub(it->second.content, it->second.content, rhs_coeff.content);
            // If the resulting coefficient is zero, remove the term
            if (mpq_equal(it->second.content, zero.content)) { lhs.erase(it); }
        } else {
            // Insert new term
            auto [pos, inserted] = lhs.emplace(rhs_key, rhs_coeff);
            RPY_DBG_ASSERT(inserted);
            mpq_neg(pos->second.content, rhs_coeff.content);
        }
    }
}

void generics::poly_mul_inplace(Polynomial& lhs, const Polynomial& rhs)
{
    Polynomial result;
    dtl::RationalCoeff zero;

    for (const auto& [lhs_key, lhs_coeff] : lhs) {
        for (const auto& [rhs_key, rhs_coeff] : rhs) {
            auto new_key = lhs_key * rhs_key;

            dtl::RationalCoeff new_coeff;
            mpq_mul(new_coeff.content, lhs_coeff.content, rhs_coeff.content);

            auto it = result.find(new_key);
            if (it != result.end()) {
                mpq_add(it->second.content,
                        it->second.content,
                        new_coeff.content);
                // If the resulting coefficient is zero, remove the term
                if (mpq_equal(it->second.content, zero.content)) {
                    result.erase(it);
                }
            } else {
                // Insert new term
                result.emplace(new_key, new_coeff);
            }
        }
    }

    // Assign result back to lhs
    lhs = std::move(result);
}

void generics::poly_div_inplace(Polynomial& lhs, mpq_srcptr rhs)
{
    dtl::RationalCoeff zero;
    if (mpq_equal(rhs, zero.content)) {
        RPY_THROW(std::domain_error, "division by zero");
    }

    for (auto it = lhs.begin(); it != lhs.end(); ++it) {
        // Divide the coefficient of the current term by rhs
        mpq_div(it->second.content, it->second.content, rhs);
    }
}
bool generics::poly_cmp_equal(
        const Polynomial& lhs,
        const Polynomial& rhs
) noexcept
{
    if (lhs.size() != rhs.size()) { return false; }

    return std::equal(
            lhs.begin(),
            lhs.end(),
            rhs.begin(),
            rhs.end(),
            [](const auto& lhs_pair, const auto& rhs_pair) {
                return lhs_pair.first == rhs_pair.first
                        && mpq_equal(
                                lhs_pair.second.content,
                                rhs_pair.second.content
                        );
            }
    );
}

hash_t generics::hash_value(const Polynomial& value)
{
    hash_t hash = 0;
    for (const auto& [key, coeff] : value) {
        hash_combine(hash, hash_value(key));
        hash_combine(hash, mpq_hash(coeff.content));
    }
    return hash;
}

void generics::poly_print(std::ostream& os, const Polynomial& value)
{
    os << '{';
    string tmp_coeff;
    for (const auto& [key, coeff] : value) {
        tmp_coeff.clear();
        mpq_display_rep(tmp_coeff, coeff.content);
        os << ' ' << tmp_coeff;
        if (RPY_UNLIKELY(!key.empty())) {
            os << '(' << key << ')';
        }
    }
    os << " }";
}

bool Polynomial::is_constant() const noexcept
{
    if (empty()) { return true; }

    if (size() == 1 && begin()->first.empty()) {
        return true;
    }

    return false;
}
