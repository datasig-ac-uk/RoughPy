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



void python::init_shuffle_tensor(py::module_ &m) {
    py::options options;
    options.disable_function_signatures();

    py::class_<ShuffleTensor> klass(m, "ShuffleTensor", SHUFFLE_TENSOR_DOC);
    klass.def(py::init(&construct_shuffle), "data"_a);

    klass.def_property_readonly("width", &ShuffleTensor::width);
    klass.def_property_readonly("depth", &ShuffleTensor::depth);
    klass.def_property_readonly("coeff_type", &ShuffleTensor::coeff_type);
    klass.def_property_readonly("storage_type", &ShuffleTensor::storage_type);

    klass.def("size", &ShuffleTensor::size);
    klass.def("dimension", &ShuffleTensor::dimension);
    klass.def("degree", &ShuffleTensor::degree);

    klass.def("__getitem__", [](const ShuffleTensor &self, key_type key) {
      return self[key];
    });
    klass.def("__iter__", [](const ShuffleTensor &self) {
          return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>());

    klass.def("__neg__", &ShuffleTensor::uminus, py::is_operator());

    klass.def("__add__", &ShuffleTensor::add, py::is_operator());
    klass.def("__sub__", &ShuffleTensor::sub, py::is_operator());
    klass.def("__mul__", &ShuffleTensor::smul, py::is_operator());
    klass.def("__truediv__", &ShuffleTensor::smul, py::is_operator());
    klass.def("__mul__", &ShuffleTensor::mul, py::is_operator());
    klass.def(
        "__rmul__", [](const ShuffleTensor &self, const scalars::Scalar &other) { return self.smul(other); },
        py::is_operator());

    klass.def(
        "__mul__", [](const ShuffleTensor &self, scalar_t arg) {
          return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const ShuffleTensor &self, scalar_t arg) {
          return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__mul__", [](const ShuffleTensor &self, long long arg) {
          return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const ShuffleTensor &self, long long arg) {
          return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const ShuffleTensor &self, scalar_t arg) {
          return self.sdiv(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const ShuffleTensor &self, long long arg) {
          return self.sdiv(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());

    klass.def("__iadd__", &ShuffleTensor::add_inplace, py::is_operator());
    klass.def("__isub__", &ShuffleTensor::sub_inplace, py::is_operator());
    klass.def("__imul__", &ShuffleTensor::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &ShuffleTensor::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &ShuffleTensor::mul_inplace, py::is_operator());

    klass.def(
        "__imul__", [](ShuffleTensor &self, scalar_t arg) {
          return self.smul_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__imul__", [](ShuffleTensor &self, long long arg) {
          return self.smul_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](ShuffleTensor &self, scalar_t arg) {
          return self.sdiv_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](ShuffleTensor &self, long long arg) {
          return self.sdiv_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());

    klass.def("add_scal_mul", &ShuffleTensor::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &ShuffleTensor::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &ShuffleTensor::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &ShuffleTensor::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &ShuffleTensor::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &ShuffleTensor::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &ShuffleTensor::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &ShuffleTensor::mul_sdiv, "other"_a, "scalar"_a);


    klass.def("__str__", [](const ShuffleTensor &self) {
      std::stringstream ss;
      self.print(ss);
      return ss.str();
    });

    klass.def("__repr__", [](const ShuffleTensor &self) {
      std::stringstream ss;
      ss << "ShuffleTensor(width=" << self.width().value()
         << ", depth=" << self.depth().value();
      ss << ", ctype=" << self.coeff_type()->info().name << ')';
      return ss.str();
    });

    klass.def("__eq__", [](const ShuffleTensor &lhs, const ShuffleTensor &rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const ShuffleTensor &lhs, const ShuffleTensor &rhs) { return lhs != rhs; });

#ifdef ROUGHPY_WITH_NUMPY
    klass.def("__array__", [](const ShuffleTensor &self) {
      //        py::dtype dtype = dtype_from(self.coeff_type());
      py::dtype dtype = python::ctype_to_npy_dtype(self.coeff_type());

      if (self.storage_type() == VectorType::Dense) {
          auto dense_data = self.dense_data().value();
          return py::array(dtype, {dense_data.size()}, {}, dense_data.ptr());
      }
      return py::array(dtype);
    });
#endif
}
