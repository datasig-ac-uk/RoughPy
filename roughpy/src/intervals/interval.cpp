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

:class:`Interval` objects are used to query a :class:`Stream` to get a :py:meth:`~signature` or a :py:meth:`~log_signature`.
They are a time-like axis which is closed on the left, open on the right, e.g. :math:`[0,1)`.

RoughPy is very careful in how it works with intervals.

One design goal is that it should be able to handle jumps in the underlying signal that occur at particular times, including the beginning or end of the :class:`Interval`, and still guarantee that if you combine the :py:meth:`~signature` over adjacent :class:`Interval`, you always get the :py:meth:`~signature` over the entire :class:`Interval`.
This implies that there has to be a decision about whether data at the exact beginning or exact end of the :class:`Interval` is included.

The convention in RoughPy are that we use clopen intervals, and that data at beginning of the :class:`Interval` is seen, and data at the end of the :class:`Interval` is seen in the next :class:`Interval`.
A second design goal is that the code should be efficient, and so the internal representation of a :class:`Stream` involves caching the :py:meth:`~signature` over dyadic intervals of different resolutions.
Recovering the :py:meth:`~signature` over any :class:`Interval` using the cache has logarithmic complexity (using at most :math:`2n` tensor multiplications, when :math:`n` is the internal resolution of the :class:`Stream`.
Resolution refers to the length of the finest granularity at which we will store information about the underlying data.

Any event occurs within one of these finest granularity intervals, multiple events occur within the same :class:`Interval` resolve to a more complex :py:meth:`~log_signature` which correctly reflects the time sequence of the events within this grain of time.
However, no query of the :class:`Stream` is allowed to see finer :py:attr:`~resolution` than the internal :py:attr:`~resolution` of the :class:`Stream`, it is only allowed to access the information over :py:class:`Interval` objects that are a union of these finest :py:attr:`~resolution` granular intervals.
For this reason, a query over any :class:`Interval` is replaced by a query is replaced by a query over an :class:`Interval` whose endpoints have been shifted to be consistent with the granular :py:attr:`~resolution`, obtained by rounding these points to the contained end-point of the unique clopen granular :class:`Interval` containing this point.
In particular, if both the left-hand and right-hand ends of the :class:`Interval` are contained in the clopen granular :class:`Interval`, we round the :class:`Interval` to the empty :class:`Interval`. Specifying a :py:attr:`~resolution` of :math:`32` or :math:`64` equates to using respective integer arithmetic.

We can create an :class:`Interval` to query a :class:`Stream` over, for example to compute a :py:meth:`~signature`, in the following way.
The example below is the interval :math:`[0,1)`, over the Reals.

.. code:: python

    interval = rp.RealInterval(0, 1)

.. note::
     Clopen is currently the only supported interval type.

)edoc";

void python::init_interval(py::module_& m)
{

    py::options options;
    options.disable_function_signatures();

    py::class_<Interval, python::PyInterval> klass(m, "Interval", INTERVAL_DOC);

    klass.def_property_readonly("interval_type", &Interval::type);

    klass.def("inf", &Interval::inf, "Returns the infimum of an :class:`Interval`.");
    klass.def("sup", &Interval::sup, "Returns the supremum of an :class:`Interval`.");
    klass.def("included_end", &Interval::included_end, "If the :class:`Interval` type is :py:attr:`clopen`, returns the infimum. If the :class:`~Interval` is :py:attr:`~opencl`, returns the supremum.");
    klass.def("excluded_end", &Interval::excluded_end, "If the :class:`Interval` type is :py:attr:`~clopen`, returns the supremum. If the :class:`~Interval` is :py:attr:`~opencl`, returns the infimum.");
    klass.def("__eq__", [](const Interval& lhs, const Interval& rhs) {
        return lhs == rhs;
    });
    klass.def("__neq__", [](const Interval& lhs, const Interval& rhs) {
        return lhs != rhs;
    });
    klass.def("intersects_with", &Interval::intersects_with, "other"_a, "Checks whether two :class:`Interval` objects overlap.");

    klass.def("contains",
              static_cast<bool (Interval::*)(param_t) const noexcept>(
                      &Interval::contains_point),
              "arg"_a);
    klass.def("contains",
              static_cast<bool (Interval::*)(const Interval&) const noexcept>(
                      &Interval::contains),
              "arg"_a, "Takes either a number, tells you if it's there or not, OR takes an :class:`Interval`, tells you if that :class:`~Interval` is fully contained in the other :class:`~Interval`.");

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
