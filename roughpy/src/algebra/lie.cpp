#include "lie.h"

#include <pybind11/operators.h>

#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/context.h>

#include "args/numpy.h"
#include "args/kwargs_to_vector_construction.h"
#include "scalars/scalars.h"

#include "lie_key.h"
#include "setup_algebra_type.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char *LIE_DOC = R"edoc(
Element of the free Lie algebra.
)edoc";

static Lie construct_lie(py::object data, py::kwargs kwargs) {
    auto helper = python::kwargs_to_construction_data(kwargs);

    python::PyToBufferOptions options;
    options.type = helper.ctype;
    options.allow_scalar = false;

    auto buffer = py_to_buffer(data, options);

    if (helper.ctype == nullptr) {
        if (options.type == nullptr) {
            throw py::value_error("could not deduce an appropriate scalar_type");
        }
        helper.ctype = options.type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size());
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            throw py::value_error("you must provide either context or both width and depth");
        }
        helper.ctx = get_context(helper.width, helper.depth, helper.ctype, {});
    }

    auto result = helper.ctx->construct_lie({std::move(buffer), helper.vtype});

    if (options.cleanup) {
        options.cleanup();
    }

    return result;
}

void python::init_lie(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<Lie> klass(m, "Lie", LIE_DOC);
    klass.def(py::init(&construct_lie), "data"_a);

    setup_algebra_type(klass);

    klass.def("__getitem__", [](const Lie &self, key_type key) {
      return self[key];
    });

    klass.def("__repr__", [](const Lie &self) {
      std::stringstream ss;
      ss << "Lie(width=" << self.width().value()
         << ", depth=" << self.depth().value();
      ss << ", ctype=" << self.coeff_type()->info().name << ')';
      return ss.str();
    });

}
