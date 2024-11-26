//
// Created by sam on 26/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MPQ_STRING_REP_H
#define ROUGHPY_GENERICS_INTERNAL_MPQ_STRING_REP_H


#include <gmp.h>

#include "roughpy/core/types.h"


namespace rpy::generics {

inline void mpq_display_rep(string& buffer, mpq_srcptr value) noexcept
{
    // The GMP docs describe the size of a mpq string representation in the
    // documentation https://gmplib.org/manual/Rational-Conversions
    auto num_size = mpz_sizeinbase(mpq_numref(value), 10);
    auto denom_size = mpz_sizeinbase(mpq_denref(value), 10);
    buffer.resize(num_size + denom_size + 3);

    mpq_get_str(buffer.data(), 10, value);

    // The buffer has at least one null byte at the end, cut these off
    while (buffer.back() == '\0') { buffer.pop_back(); }
}

}

#endif //ROUGHPY_GENERICS_INTERNAL_MPQ_STRING_REP_H
