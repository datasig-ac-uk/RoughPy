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

#include "dyadic_interval.h"

#include <pybind11/stl.h>

#include <roughpy/intervals/dyadic.h>
#include <roughpy/intervals/dyadic_interval.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char* DYADIC_INTERVAL_DOC = R"edoc(A dyadic interval.
)edoc";

static const char* TO_DYADIC_INT_DOC
        = R"edoc(Dissect an interval into a partition of dyadic intervals.
)edoc";

void python::init_dyadic_interval(py::module_& m)
{
    using multiplier_t = typename Dyadic::multiplier_t;
    using power_t = typename Dyadic::power_t;

    py::options options;
    options.disable_function_signatures();

    py::class_<DyadicInterval, Interval> klass(m, "DyadicInterval",
                                               DYADIC_INTERVAL_DOC);

    klass.def(py::init<>());
    klass.def(py::init<IntervalType>(), "interval_type"_a);
    klass.def(py::init<multiplier_t>(), "k"_a);
    klass.def(py::init<multiplier_t, power_t>(), "k"_a, "n"_a);
    klass.def(py::init<multiplier_t, power_t, IntervalType>(), "k"_a, "n"_a,
              "interval_type"_a);
    klass.def(py::init<Dyadic>(), "dyadic"_a);
    klass.def(py::init<Dyadic, power_t>(), "dyadic"_a, "resolution"_a);

    klass.def("dyadic_included_end", &DyadicInterval::dincluded_end);
    klass.def("dyadic_excluded_end", &DyadicInterval::dexcluded_end);
    klass.def("dyadic_inf", &DyadicInterval::dinf);
    klass.def("dyadic_sup", &DyadicInterval::dsup);

    klass.def("shrink_to_contained_end",
              &DyadicInterval::shrink_to_contained_end, "arg"_a = 1);
    klass.def("shrink_to_omitted_end", &DyadicInterval::shrink_to_omitted_end);
    klass.def("shrink_left", &DyadicInterval::shrink_interval_left);
    klass.def("shrink_right", &DyadicInterval::shrink_interval_right);

    klass.def_static("to_dyadic_intervals", &to_dyadic_intervals, "interval"_a,
                     "resolution"_a, "interval_type"_a, TO_DYADIC_INT_DOC);
    klass.def_static(
            "to_dyadic_intervals",
            [](param_t inf, param_t sup, power_t resolution,
               IntervalType itype) {
                return to_dyadic_intervals(RealInterval(inf, sup), resolution,
                                           itype);
            },
            "inf"_a, "sup"_a, "resolution"_a, "interval_type"_a);
}
