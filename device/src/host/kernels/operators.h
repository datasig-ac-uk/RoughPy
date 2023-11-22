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

//
// Created by user on 19/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_CPU_KERNELS_OPERATORS_H_
#define ROUGHPY_DEVICE_SRC_CPU_KERNELS_OPERATORS_H_

#include <algorithm>
#include <utility>

namespace rpy {
namespace devices {
namespace kernels {

struct identity {

    template <typename T>
    constexpr T operator()(T arg) const noexcept
    {
        return arg;
    }
};

struct uminus {

    template <typename T>
    constexpr T operator()(T arg) const noexcept
    {
        return -arg;
    }
};

template <typename S>
struct pre_multiply {
    S m_value;

    constexpr pre_multiply(S&& val) : m_value(std::move(val)) {}

    template <typename T>
    constexpr S operator()(const T& arg) const
    {
        return m_value * arg;
    }
};

template <typename S>
struct post_multiply {
    S m_value;

    constexpr post_multiply(S&& val) : m_value(std::move(val)) {}

    template <typename T>
    constexpr S operator()(const T& arg) const
    {
        return arg * m_value;
    }
};

struct pair_max {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const noexcept
    {
        return (lhs <= rhs) ? rhs : lhs;
    }
};

struct pair_min {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const noexcept
    {
        return (lhs <= rhs) ? lhs : rhs;
    }
};

struct plus {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const
    {
        return lhs + rhs;
    }
};

struct minus {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const
    {
        return lhs - rhs;
    }
};

struct multiply {
    template <typename T>
    constexpr T operator()(const T& lhs, const T& rhs) const
    {
        return lhs * rhs;
    }
};

struct divide {
    template <typename T, typename R>
    constexpr T operator()(const T& num, const R& denom) const
    {
        return num / denom;
    }
};

}// namespace kernels
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPU_KERNELS_OPERATORS_H_
