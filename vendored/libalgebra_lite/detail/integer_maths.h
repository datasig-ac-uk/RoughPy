//
// Created by user on 13/07/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_INTEGER_MATHS_H_
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_INTEGER_MATHS_H_

#include "macros.h"
#include "traits.h"

namespace lal {

template <typename I>
constexpr enable_if_t<is_integral<I>::value, bool> is_even(I integer) noexcept
{
    return (integer & 1) == 0;
}


}


#endif// LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_DETAIL_INTEGER_MATHS_H_
