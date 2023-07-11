// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

//
// Created by user on 04/07/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_INTERVALS_DATE_TIME_INTERVAL_H_
#define ROUGHPY_ROUGHPY_SRC_INTERVALS_DATE_TIME_INTERVAL_H_

#include <roughpy/intervals/interval.h>

#include "args/convert_timestamp.h"
#include "roughpy_module.h"

namespace rpy {
namespace python {

class DateTimeInterval : public intervals::Interval
{
    py::object m_dt_begin;
    py::object m_dt_end;

public:
    DateTimeInterval(py::object dt_begin, py::object dt_end);

    // It will be unusual for these methods to be used to get the
    // inf and sup of the parameter interval.
    param_t inf() const override;
    param_t sup() const override;

    RPY_NO_DISCARD
    py::object dt_inf() const { return m_dt_begin; }
    RPY_NO_DISCARD
    py::object dt_sup() const { return m_dt_end; };
};

void init_datetime_interval(py::module_& m);


}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_INTERVALS_DATE_TIME_INTERVAL_H_
