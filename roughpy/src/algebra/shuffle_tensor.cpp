#include "shuffle_tensor.h"

#include <sstream>

#include <pybind11/operators.h>

#include <roughpy/algebra/shuffle_tensor.h>
#include <roughpy/algebra/context.h>

#include "args/numpy.h"
#include "args/kwargs_to_vector_construction.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"

#include "tensor_key.h"
#include "setup_algebra_type.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;


static const char* SHUFFLE_TENSOR_DOC = R"eadoc(Element of the shuffle tensor algebra.
)eadoc";

static ShuffleTensor construct_shuffle(py::object data, py::kwargs kwargs) {
    auto helper = python::kwargs_to_construction_data(kwargs);

    auto py_key_type = py::type::of<python::PyTensorKey>();
    python::AlternativeKeyType alt{
        py_key_type,
        [](py::handle py_key) -> key_type {
            return static_cast<key_type>(py_key.cast<python::PyTensorKey>());
        }};

    python::PyToBufferOptions options;
    options.type = helper.ctype;
    options.alternative_key = &alt;

    auto buffer = python::py_to_buffer(data, options);

    if (helper.ctype == nullptr) {
        if (options.type == nullptr) {
            throw py::value_error("could not deduce appropriate scalar type");
        }
        helper.ctype = options.type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size()) - 1;
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            throw py::value_error("you must provide either context or both width and depth");
        }
        helper.ctx = get_context(helper.width, helper.depth, helper.ctype, {});
    }

    if (!helper.vtype_requested) {
        if (buffer.has_keys()) {
            // if data comes and k-v pairs, then it is reasonable to assume
            // the user wants a sparse tensor.
            helper.vtype = VectorType::Sparse;
        } else {
            // otherwise dense
            helper.vtype = VectorType::Dense;
        }
    }

    auto result = helper.ctx->construct_shuffle_tensor({std::move(buffer), helper.vtype});

    if (options.cleanup) {
        options.cleanup();
    }

    return result;
}



void rpy::python::init_shuffle_tensor(py::module_ &m) {
    py::options options;
    options.disable_function_signatures();

    py::class_<ShuffleTensor> klass(m, "ShuffleTensor", SHUFFLE_TENSOR_DOC);
    klass.def(py::init(&construct_shuffle), "data"_a);
    setup_algebra_type(klass);

    klass.def("__getitem__", [](const ShuffleTensor &self, key_type key) {
      return self[key];
    });
    klass.def("__iter__", [](const ShuffleTensor &self) {
          return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>());

    klass.def("__repr__", [](const ShuffleTensor &self) {
      std::stringstream ss;
      ss << "ShuffleTensor(width=" << self.width().value()
         << ", depth=" << self.depth().value();
      ss << ", ctype=" << self.coeff_type()->info().name << ')';
      return ss.str();
    });


}
