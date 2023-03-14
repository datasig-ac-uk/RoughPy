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

    klass.def_property_readonly("width", &Lie::width);
    klass.def_property_readonly("max_degree", &Lie::depth);
    klass.def_property_readonly("dtype", &Lie::coeff_type);
    klass.def_property_readonly("storage_type", &Lie::storage_type);

    klass.def("size", &Lie::size);
    klass.def("degree", &Lie::degree);

    klass.def("__getitem__", [](const Lie &self, key_type key) {
      return self[key];
    });
    klass.def("__iter__", [](const Lie &self) {
      return py::make_iterator(self.begin(), self.end());
    });

    klass.def("__neg__", &Lie::uminus, py::is_operator());

    klass.def("__add__", &Lie::add, py::is_operator());
    klass.def("__sub__", &Lie::sub, py::is_operator());
    klass.def("__mul__", &Lie::smul, py::is_operator());
    klass.def("__truediv__", &Lie::smul, py::is_operator());
    klass.def("__mul__", &Lie::mul, py::is_operator());
    klass.def(
        "__rmul__", [](const Lie &self, const scalars::Scalar &other) { return self.smul(other); },
        py::is_operator());

    klass.def(
        "__mul__", [](const Lie &self, scalar_t arg) {
          return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__mul__", [](const Lie &self, long long arg) {
          return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const Lie &self, scalar_t arg) {
          return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const Lie &self, long long arg) {
          return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const Lie &self, scalar_t arg) {
          return self.sdiv(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const Lie &self, scalar_t arg) {
          return self.sdiv(scalars::Scalar(self.coeff_type(), arg));
        },
        py::is_operator());

    klass.def("__iadd__", &Lie::add_inplace, py::is_operator());
    klass.def("__isub__", &Lie::sub_inplace, py::is_operator());
    klass.def("__imul__", &Lie::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &Lie::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &Lie::mul_inplace, py::is_operator());

    klass.def(
        "__imul__", [](Lie &self, scalar_t arg) {
          return self.smul_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__imul__", [](Lie &self, long long arg) {
          return self.smul_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](Lie &self, scalar_t arg) {
          return self.sdiv_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](Lie &self, long long arg) {
          return self.sdiv_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());

    klass.def("add_scal_mul", &Lie::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &Lie::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &Lie::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &Lie::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &Lie::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &Lie::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &Lie::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &Lie::mul_sdiv, "other"_a, "scalar"_a);

    klass.def("__str__", [](const Lie &self) {
      std::stringstream ss;
      self.print(ss);
      return ss.str();
    });

    klass.def("__repr__", [](const Lie &self) {
      std::stringstream ss;
      ss << "Lie(width=" << self.width().value()
         << ", depth=" << self.depth().value();
      ss << ", ctype=" << self.coeff_type()->info().name << ')';
      return ss.str();
    });

    klass.def("__eq__", [](const Lie &lhs, const Lie &rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const Lie &lhs, const Lie &rhs) { return lhs != rhs; });

#ifdef ROUGHPY_WITH_NUMPY
    klass.def("__array__", [](const Lie &arg) {
      py::dtype dtype = python::ctype_to_npy_dtype(arg.coeff_type());

      if (arg.storage_type() == VectorType::Dense) {
          auto dense_data = arg.dense_data().value();
          return py::array(dtype, {dense_data.size()}, {}, dense_data.ptr());
      }
      return py::array(dtype);
    });
#endif
}
