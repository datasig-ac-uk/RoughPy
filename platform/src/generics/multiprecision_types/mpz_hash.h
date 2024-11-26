//
// Created by sammorley on 25/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MPZ_HASH_H
#define ROUGHPY_GENERICS_INTERNAL_MPZ_HASH_H

#include <gmp.h>

#include <roughpy/core/hash.h>

namespace rpy::generics {


inline hash_t mpz_hash(mpz_srcptr integer) noexcept
{
    const size_t nlimbs = mpz_size(integer);
    const auto* limbs = mpz_limbs_read(integer);

    hash_t result = Hash<mp_size_t>{}(integer->_mp_size);
    for (size_t i = 0; i < nlimbs; ++i) {
        hash_combine(result, limbs[i]);
    }
    return result;
}

inline hash_t mpq_hash(mpq_srcptr integer) noexcept
{
    auto num_hash = mpz_hash(mpq_numref(integer));
    const auto denom_hash = mpz_hash(mpq_denref(integer));

    hash_combine(num_hash, denom_hash);
    return num_hash;
}


}

#endif //ROUGHPY_GENERICS_INTERNAL_MPZ_HASH_H
