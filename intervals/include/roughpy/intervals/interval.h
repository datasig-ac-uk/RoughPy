// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_INTERVALS_INTERVAL_H_
#define ROUGHPY_INTERVALS_INTERVAL_H_

#include <roughpy/core/types.h>

#include <iosfwd>

namespace rpy {
namespace intervals {

enum class IntervalType
{
    Clopen,
    Opencl
};

class RPY_EXPORT Interval
{
protected:
    IntervalType m_interval_type = IntervalType::Clopen;

public:
    Interval() = default;

    explicit Interval(IntervalType itype) : m_interval_type(itype) {}

    virtual ~Interval() = default;

    RPY_NO_DISCARD
    inline IntervalType type() const noexcept { return m_interval_type; }

    RPY_NO_DISCARD
    virtual param_t inf() const = 0;
    RPY_NO_DISCARD
    virtual param_t sup() const = 0;

    RPY_NO_DISCARD
    virtual param_t included_end() const;
    RPY_NO_DISCARD
    virtual param_t excluded_end() const;

    RPY_NO_DISCARD
    virtual bool contains_point(param_t arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool is_associated(const Interval& arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool contains(const Interval& arg) const noexcept;
    RPY_NO_DISCARD
    virtual bool intersects_with(const Interval& arg) const noexcept;

    RPY_NO_DISCARD
    virtual bool operator==(const Interval& other) const;
    RPY_NO_DISCARD
    virtual bool operator!=(const Interval& other) const;
};

RPY_EXPORT
std::ostream& operator<<(std::ostream& os, const Interval& interval);

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_INTERVAL_H_
