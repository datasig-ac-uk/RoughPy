// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_CORE_POINTER_HELPERS_H_
#define ROUGHPY_CORE_POINTER_HELPERS_H_

#include "macros.h"
#include "traits.h"
#include "types.h"
#include "detail/bit_cast.h" // IWYU pragma: EXPORT

#include <bitset>
#include <climits>
#include <cstring>
#include <iterator>
#include <vector>

namespace rpy {



template <typename T>
RPY_NO_DISCARD inline enable_if_t<is_integral_v<T>, size_t>
count_bits(T val) noexcept
{
    // There might be optimisations using builtin popcount functions
    return std::bitset<CHAR_BIT * sizeof(T)>(val).count();
}

RPY_NO_DISCARD constexpr size_t static_log2p1(size_t value) noexcept
{
    return (value == 0) ? 0 : 1 + static_log2p1(value >> 1);
}

/**
 * @brief
 * @tparam T
 */
template <typename T>
class MaybeOwned
{
    enum State
    {
        IsOwned,
        IsBorrowed
    };

    T* p_data;
    State m_state;

public:
    constexpr MaybeOwned(std::nullptr_t) : p_data(nullptr), m_state(IsOwned) {}
    constexpr MaybeOwned(T* ptr) : p_data(ptr), m_state(IsBorrowed) {}

    ~MaybeOwned()
    {
        if (m_state == IsOwned) { delete[] p_data; }
    }

    constexpr MaybeOwned& operator=(T* ptr)
    {
        p_data = ptr;
        m_state = IsOwned;
        return *this;
    }

    RPY_NO_DISCARD operator T*() const noexcept { return p_data; }

    RPY_NO_DISCARD operator bool() const noexcept { return p_data != nullptr; }
};



template <typename I, typename J>
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<I>, I>
round_up_divide(I value, J divisor) noexcept {
    return (value + static_cast<I>(divisor) - 1) / static_cast<I>(divisor);
}


template <typename I>
RPY_NO_DISCARD constexpr enable_if_t<is_integral_v<I>, I>
next_power_2(I value, I start=I(1)) noexcept
{
    if (value == 0) { return 0; }
    if (is_signed_v<I> && value < 0) { return -next_power_2(-value); }
    return (start >= value) ? start : next_power_2(value, I(start << 1));
}


}// namespace rpy

#endif// ROUGHPY_CORE_POINTER_HELPERS_H_
