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

    hash_t result = 0;
    for (size_t i = 0; i < nlimbs; ++i) {
        hash_combine(result, limbs[i]);
    }
    return result;
}


}

#endif //ROUGHPY_GENERICS_INTERNAL_MPZ_HASH_H
