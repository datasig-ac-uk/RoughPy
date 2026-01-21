//
// Created by sam on 19/11/24.
//

#ifndef ROUGHPY_CORE_DETAIL_BIT_CAST_H
#define ROUGHPY_CORE_DETAIL_BIT_CAST_H

#include <cstring>
#include <type_traits>
#include <utility>

#include "config.h"

namespace rpy {


/**
 * @brief Cast the bit value from a value of type From to a value
 * of type To.
 *
 * The implementation here is pretty similar to the example from
 * https://en.cppreference.com/w/cpp/numeric/bit_cast
 * if we have to define our own version using memcpy.
 */
#if defined(__cpp_lib_bit_cast) && __cpp_Lib_bit_cast >= 201806L
using std::bit_cast;
#else
template <typename To, typename From>
[[nodiscard]] std::enable_if_t<
        sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From>
                && std::is_trivially_copyable_v<To>
                && std::is_default_constructible_v<To>,
        To>
bit_cast(From from)
{
    To to;
    std::memcpy(static_cast<void*>(std::addressof(to)),
           static_cast<const void*>(std::addressof(from)), sizeof(To));
    return to;
}
#endif

}

#endif //ROUGHPY_CORE_DETAIL_BIT_CAST_H
