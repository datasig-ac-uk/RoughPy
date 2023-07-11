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

#include "real_interval.h"

#include <sstream>

#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char* REAL_INTERVAL_DOC
        = R"edoc(A half-open interval in the real line.
)edoc";

void python::init_real_interval(py::module_& m)
{

    py::class_<RealInterval, Interval> klass(m, "RealInterval",
                                             REAL_INTERVAL_DOC);
    //    klass.def(py::init<>());
    //    klass.def(py::init<IntervalType>());
    klass.def(py::init<double, double>(), "inf"_a, "sup"_a);
    klass.def(py::init<double, double, IntervalType>(), "inf"_a, "sup"_a,
              "interval_type"_a);
    klass.def(py::init<const Interval&>(), "arg"_a);
    klass.def("__repr__", [](const RealInterval& arg) {
        std::stringstream ss;
        ss << "RealInterval(inf=" << std::to_string(arg.inf())
           << ", sup=" << std::to_string(arg.sup()) << ", type=";
        if (arg.type() == IntervalType::Clopen) {
            ss << "clopen";
        } else {
            ss << "opencl";
        }
        ss << ')';
        return ss.str();
    });
}
