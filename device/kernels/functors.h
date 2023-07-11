// Copyright (c) 2023 Datasig Developers. All rights reserved.
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

#ifndef ROUGHPY_DEVICE_FUNCTORS_H_
#define ROUGHPY_DEVICE_FUNCTORS_H_

#include "roughpy/device/core.h"

#if defined(__NVCC__)
#  include "../../../../../../usr/include/thrust/functional.h"
#elif defined(__HIP__)
#  include <rocThrust/functional.h>
#else
#  include <functional>
#endif

namespace rpy {
namespace device {

#if defined(__NVCC__) || defined(__HIP__)

using UMinus = thrust::negate<>;
using Add = thrust::plus<>;
using Sub = thrust::minus<>;
using Mul = thrust::multiplies<>;
using Div = thrust::divides<>;

#else

using UMinus = std::negate<>;
using Add = std::plus<>;
using Sub = std::minus<>;
using Mul = std::multiplies<>;
using Div = std::divides<>;

#endif

template <typename S>
class LeftScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit LeftScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S operator()(const S& x) const noexcept
    {
        return m_val * x;
    }
};

template <typename S>
class RightScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit RightScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S operator()(const S& x) const noexcept
    {
        return x * m_val;
    }
};

template <typename S>
class AddLeftScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit AddLeftScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S
    operator()(const S& x, const S& y) const noexcept
    {
        return x + m_val * y;
    }
};

template <typename S>
class SubLeftScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit SubLeftScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S
    operator()(const S& x, const S& y) const noexcept
    {
        return x - m_val * y;
    }
};

template <typename S>
class AddRightScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit AddRightScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S
    operator()(const S& x, const S& y) const noexcept
    {
        return x + y * m_val;
    }
};

template <typename S>
class SubRightScalarMul
{
    S m_val;

public:
    RPY_DEVICE_HOST explicit SubRightScalarMul(S value) : m_val(value) {}

    RPY_DEVICE_HOST RPY_STRONG_INLINE S
    operator()(const S& x, const S& y) const noexcept
    {
        return x - y * m_val;
    }
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_FUNCTORS_H_
