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
// Created by user on 20/06/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_ARGS_CONVERT_TIMESTAMP_H
#define ROUGHPY_ROUGHPY_SRC_ARGS_CONVERT_TIMESTAMP_H

// #define PY_SSIZE_T_CLEAN
// #include <Python.h>

#include "roughpy_module.h"

namespace rpy {
namespace python {

bool is_py_datetime(py::handle object) noexcept;
bool is_py_date(py::handle object) noexcept;
bool is_py_time(py::handle object) noexcept;

enum class PyDateTimeResolution : uint8_t
{
    Microseconds = 0,
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
    Days
};

struct PyDateTimeConversionOptions {
    PyDateTimeResolution resolution;
};

void init_datetime(py::module_& m);

param_t convert_delta_from_datetimes(
        py::handle current, py::handle previous,
        const PyDateTimeConversionOptions& options
);
param_t convert_timedelta(
        py::handle py_timedelta, const PyDateTimeConversionOptions& options
);

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_ARGS_CONVERT_TIMESTAMP_H
