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

#include "dyadic.h"

#include <pybind11/operators.h>
#include <roughpy/intervals/dyadic.h>
#include <sstream>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char* DYADIC_DOC = R"edoc(A dyadic rational number.
)edoc";

void python::init_dyadic(py::module_& m)
{
    using multiplier_t = typename Dyadic::multiplier_t;
    using power_t = typename Dyadic::power_t;

    py::class_<Dyadic> klass(m, "Dyadic", DYADIC_DOC);

    klass.def(py::init<>());
    klass.def(py::init<multiplier_t>(), "k"_a);
    klass.def(py::init<multiplier_t, power_t>(), "k"_a, "n"_a);

    klass.def("__float__", [](const Dyadic& dia) {
        return static_cast<param_t>(dia);
    });

    klass.def("rebase", &Dyadic::rebase, "resolution"_a);
    klass.def("__str__", [](const Dyadic& dia) {
        std::stringstream ss;
        ss << dia;
        return ss.str();
    });
    klass.def("__repr__", [](const Dyadic& dia) {
        std::stringstream ss;
        ss << "Dyadic" << dia;
        return ss.str();
    });

    klass.def_static("dyadic_equals", &dyadic_equals, "lhs"_a, "rhs"_a);
    klass.def_static("rational_equals", &rational_equals, "lhs"_a, "rhs"_a);

    klass.def_property_readonly("k", &Dyadic::multiplier);
    klass.def_property_readonly("n", &Dyadic::power);

    klass.def(py::self < py::self);
    klass.def(py::self <= py::self);
    klass.def(py::self > py::self);
    klass.def(py::self >= py::self);

    klass.def("__iadd__", [](Dyadic& dia, multiplier_t val) {
        return dia.move_forward(val);
    });
}
