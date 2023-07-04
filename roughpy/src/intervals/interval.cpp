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

#include "interval.h"

#include <sstream>

#include <roughpy/intervals/interval.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char* INTERVAL_DOC = R"edoc(
Half-open interval in the real line.
)edoc";

void python::init_interval(py::module_& m)
{

    py::options options;
    options.disable_function_signatures();

    py::class_<Interval, python::PyInterval> klass(m, "Interval", INTERVAL_DOC);

    klass.def_property_readonly("interval_type", &Interval::type);

    klass.def("inf", &Interval::inf);
    klass.def("sup", &Interval::sup);
    klass.def("included_end", &Interval::included_end);
    klass.def("excluded_end", &Interval::excluded_end);
    klass.def("__eq__", [](const Interval& lhs, const Interval& rhs) {
        return lhs == rhs;
    });
    klass.def("__neq__", [](const Interval& lhs, const Interval& rhs) {
        return lhs != rhs;
    });
    klass.def("intersects_with", &Interval::intersects_with, "other"_a);

    klass.def("contains",
              static_cast<bool (Interval::*)(param_t) const noexcept>(
                      &Interval::contains_point),
              "arg"_a);
    klass.def("contains",
              static_cast<bool (Interval::*)(const Interval&) const noexcept>(
                      &Interval::contains),
              "arg"_a);

    klass.def("__repr__", [](const Interval& arg) {
        std::stringstream ss;
        ss << "Interval(inf=" << std::to_string(arg.inf())
           << ", sup=" << std::to_string(arg.sup()) << ", type=";
        if (arg.type() == IntervalType::Clopen) {
            ss << "clopen";
        } else {
            ss << "opencl";
        }
        ss << ')';
        return ss.str();
    });
    klass.def("__str__", [](const Interval& arg) {
        std::stringstream ss;
        ss << arg;
        return ss.str();
    });
}

param_t python::PyInterval::inf() const
{
    PYBIND11_OVERRIDE_PURE(param_t, Interval, inf);
}
param_t python::PyInterval::sup() const
{
    PYBIND11_OVERRIDE_PURE(param_t, Interval, sup);
}
param_t python::PyInterval::included_end() const
{
    PYBIND11_OVERRIDE(param_t, Interval, included_end);
}
param_t python::PyInterval::excluded_end() const
{
    PYBIND11_OVERRIDE(param_t, Interval, excluded_end);
}
bool python::PyInterval::contains_point(param_t arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, Interval, contains_point, arg);
}
bool python::PyInterval::is_associated(const Interval& arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, Interval, is_associated, arg);
}
bool python::PyInterval::contains(const Interval& arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, Interval, contains, arg);
}
bool python::PyInterval::intersects_with(const Interval& arg) const noexcept
{
    PYBIND11_OVERRIDE(bool, Interval, intersects_with, arg);
}
