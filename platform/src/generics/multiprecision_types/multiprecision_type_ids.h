//
// Created by sam on 28/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_TYPE_IDS_H
#define ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_TYPE_IDS_H


#include <gmp.h>
#include <mpfr.h>

#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "generics/builtin_types/builtin_type_ids.h"

namespace rpy::generics {

using MPInt = remove_pointer_t<mpz_ptr>;
using MPRational = remove_pointer_t<mpq_ptr>;
using MPFloat = remove_pointer_t<mpfr_ptr>;


template <>
inline constexpr string_view type_id_of<MPInt> = "apz";

template <>
inline constexpr string_view type_id_of<MPRational> = "apq";

template <>
inline constexpr string_view type_id_of<MPFloat> = "apf";



}

#endif //ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_TYPE_IDS_H
