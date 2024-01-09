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

#ifndef ROUGHPY_STREAMS_PARAMETRIZATION_H_
#define ROUGHPY_STREAMS_PARAMETRIZATION_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>
#include <roughpy/platform/serialization.h>

#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {

class ROUGHPY_STREAMS_EXPORT Parameterization
{
    param_t m_param_offset = 0.0;
    param_t m_param_scaling = 1.0;

    bool b_add_to_channels = false;
    idimn_t m_is_channel = -1;

public:
    virtual ~Parameterization();

    void add_as_channel() noexcept { b_add_to_channels = true; }

    RPY_NO_DISCARD bool is_channel() const noexcept
    {
        return b_add_to_channels || (m_is_channel >= 0);
    }

    RPY_NO_DISCARD bool needs_adding() const noexcept
    {
        return b_add_to_channels && m_is_channel < 0;
    }

    RPY_NO_DISCARD intervals::RealInterval
    reparametrize(const intervals::RealInterval& arg) const
    {
        return {m_param_offset + m_param_scaling * arg.inf(),
                m_param_offset + m_param_scaling * arg.sup(), arg.type()};
    }

    RPY_NO_DISCARD virtual param_t start_param() const noexcept;

    RPY_NO_DISCARD virtual intervals::RealInterval
    convert_parameter_interval(const intervals::Interval& arg) const;
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_PARAMETRIZATION_H_
