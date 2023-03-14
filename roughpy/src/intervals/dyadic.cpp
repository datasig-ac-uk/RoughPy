#include "dyadic.h"

#include <sstream>
#include <pybind11/operators.h>
#include <roughpy/intervals/dyadic.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;

static const char *DYADIC_DOC = R"edoc(A dyadic rational number.
)edoc";


void python::init_dyadic(py::module_ &m) {
    using multiplier_t = typename Dyadic::multiplier_t;
    using power_t = typename Dyadic::power_t;

    py::class_<Dyadic> klass(m, "Dyadic", DYADIC_DOC);

    klass.def(py::init<>());
    klass.def(py::init<multiplier_t>(), "k"_a);
    klass.def(py::init<multiplier_t, power_t>(), "k"_a, "n"_a);

    klass.def("__float__", [](const Dyadic &dia) { return static_cast<param_t>(dia); });

    klass.def("rebase", &Dyadic::rebase, "resolution"_a);
    klass.def("__str__", [](const Dyadic &dia) {
        std::stringstream ss;
        ss << dia;
        return ss.str();
    });
    klass.def("__repr__", [](const Dyadic &dia) {
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

    klass.def("__iadd__", [](Dyadic &dia, multiplier_t val) { return dia.move_forward(val); });
}
