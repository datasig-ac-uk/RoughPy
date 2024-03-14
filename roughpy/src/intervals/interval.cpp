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

Intervals are used to query a stream to get a signature or a log signature.
They are a time-like axis which is closed on the left, open on the right, e.g. [0,1).

RoughPy is very careful in how it works with intervals.

One design goal is that it should be able to handle jumps in the underlying signal that occur at particular times, including the beginning or end of the interval, and still guarantee that if you combine the signature over adjacent interval, you always get the signature over the entire interval.
This implies that there has to be a decision about whether data at the exact beginning or exact end of the interval is included.

The convention in RoughPy are that we use clopen intervals, and that data at beginning of the interval is seen, and data at the end of the interval is seen in the next interval.
A second design goal is that the code should be efficient, and so the internal representation of a stream involves caching the signature over dyadic intervals of different resolutions.
Recovering the signature over any interval using the cache has logarithmic complexity (using at most 2n tensor multiplications, when n is the internal resolution of the stream).
Resolution refers to the length of the finest granularity at which we will store information about the underlying data.

Any event occurs within one of these finest granularity intervals, multiple events occur within the same interval resolve to a more complex log-signature which correctly reflects the time sequence of the events within this grain of time.
However, no query of the stream is allowed to see finer resolution than the internal resolution of the stream, it is only allowed to access the information over intervals that are a union of these finest resolution granular intervals.
For this reason, a query over any interval is replaced by a query is replaced by a query over an interval whose endpoints have been shifted to be consistent with the granular resolution, obtained by rounding these points to the contained end-point of the unique clopen granular interval containing this point.
In particular, if both the left-hand and right-hand ends of the interval are contained in the clopen granular interval, we round the interval to the empty interval. Specifying a resolution of 32 or 64 equates to using integer arithmetic.

We can create an interval to query a stream over, for example to compute a signature, in the following way.
The example below is the interval [0,1), over the Reals.

.. code:: python
    interval = rp.RealInterval(0, 1)

Note: Clopen is currently the only supported interval type.
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
