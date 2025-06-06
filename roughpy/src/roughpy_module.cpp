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

//
// Created by user on 11/03/23.
//
// Python header first

#include "roughpy_python.h"

#include "roughpy_module.h"

#include "algebra/algebra.h"
#include "args/convert_timestamp.h"
#include "intervals/intervals.h"
#include "scalars/scalars.h"
#include "streams/streams.h"

#include "args/numpy.h"


#ifndef ROUGHPY_VERSION_STRING
#  define ROUGHPY_VERSION_STRING "1.0.0"
#endif




PYBIND11_MODULE(_roughpy, m)
{
    using namespace rpy::python;

    init_roughpy_module(m);
}


void rpy::python::init_roughpy_module(py::module_& m)
{
    m.add_object("__version__", py::str(ROUGHPY_VERSION_STRING));
    init_datetime(m);

    init_scalars(m);
    init_intervals(m);
    init_algebra(m);
    init_streams(m);
    //    init_recombine(m);


    import_numpy();
}
