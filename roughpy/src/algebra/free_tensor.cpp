#include "free_tensor.h"

#include <sstream>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/scalars/scalar.h>

#include "args/numpy.h"
#include "args/kwargs_to_vector_construction.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"

#include "tensor_key.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char *FREE_TENSOR_DOC = R"eadoc(Element of the (truncated) tensor algebra.

A :class:`tensor` object supports arithmetic operators, providing both objects are compatible,
along with comparison operators. The multiplication operator for this class is the free tensor
multiplication (concatenation of tensor words). Moreover, :class:`tensor` objects are
`iterable <https://docs.python.org/3/glossary.html#term-iterable>`_, where the items are tuples
of :class:`tensor_key` and :class:`float` corresponding to the non-zero elements of the
:class:`tensor`.

The class also supports (implicit and explicit) conversion to a Numpy array type, so it can be
used as an argument to any function that takes Numpy arrays. The array representation of a
:class:`tensor` is one-dimensional. Alternatively, one can construct d-dimensional arrays
containing the elements of degree d by using the :py:meth:`~tensor.degree_array` method.

There are methods for computing the tensor :py:meth:`~tensor.exponential`,
:py:meth:`~tensor.logarithm`, and :py:meth:`~tensor.inverse` (of group-like elements). See the
documentation of these methods for more information.

A tensor can be created from an array-like object containing the coefficients of the keys, in
their standard order. Since tensors must be created with both an alphabet size and depth, we need
to provide at least the ``depth`` argument. However, it is recommended that you also provided
the ``width`` argument, otherwise it is assumed that the tensor has degree 1 and the alphabet
size will be determined from the length of the argument.

.. code:: python

    >>> ts1 = esig_paths.tensor([1.0, 2.0, 3.0], depth=2)
    >>> print(ts1)
    { 1() 2(1) 3(2) }
    >>> ts2 = esig_paths.tensor([1.0, 2.0, 3.0], width=2, depth=2)
    >>> print(ts2)
    { 1() 2(1) 3(2) }

If the width argument is provided, this construction can be used to construct tensors of any
degree, up to the maximum. The :class:`~esig_paths.algebra_context` class provides a method
:py:meth:`~esig_paths.algebra_context.tensor_size` that can be used to get the dimension of the
tensor algebra_old up to a given degree.
)eadoc";

static FreeTensor construct_free_tensor(py::object data, py::kwargs kwargs) {
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

    auto result = helper.ctx->construct_free_tensor({std::move(buffer), helper.vtype});

    if (options.cleanup) {
        options.cleanup();
    }

    return result;
}

void python::init_free_tensor(py::module_ &m) {

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<FreeTensor> klass(m, "FreeTensor", FREE_TENSOR_DOC);
    klass.def(py::init(&construct_free_tensor), "data"_a);

    klass.def_property_readonly("width", &FreeTensor::width);
    klass.def_property_readonly("max_degree", &FreeTensor::depth);
    klass.def_property_readonly("dtype", [](const FreeTensor &arg) { return python::to_ctype_type(arg.coeff_type()); });
    klass.def_property_readonly("storage_type", &FreeTensor::storage_type);

    klass.def("size", &FreeTensor::size);
    klass.def("degree", &FreeTensor::degree);

    klass.def("__getitem__", [](const FreeTensor &self, key_type key) {
        return self[key];
    });
    klass.def(
        "__iter__", [](const FreeTensor &self) {
            return py::make_iterator(self.begin(), self.end());
        },
        py::keep_alive<0, 1>());

    klass.def("__neg__", &FreeTensor::uminus, py::is_operator());

    klass.def("__add__", &FreeTensor::add, py::is_operator());
    klass.def("__sub__", &FreeTensor::sub, py::is_operator());
    klass.def("__mul__", &FreeTensor::smul, py::is_operator());
    klass.def("__truediv__", &FreeTensor::smul, py::is_operator());
    klass.def("__mul__", &FreeTensor::mul, py::is_operator());
    klass.def(
        "__rmul__", [](const FreeTensor &self, const scalars::Scalar &other) { return self.smul(other); },
        py::is_operator());

    klass.def(
        "__mul__", [](const FreeTensor &self, scalar_t arg) {
            return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const FreeTensor &self, scalar_t arg) {
            return self.smul(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__mul__", [](const FreeTensor &self, long long arg) {
            return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__rmul__", [](const FreeTensor &self, long long arg) {
            return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const FreeTensor &self, scalar_t arg) {
            return self.sdiv(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__truediv__", [](const FreeTensor &self, long long arg) {
            return self.sdiv(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());

    klass.def("__iadd__", &FreeTensor::add_inplace, py::is_operator());
    klass.def("__isub__", &FreeTensor::sub_inplace, py::is_operator());
    klass.def("__imul__", &FreeTensor::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &FreeTensor::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &FreeTensor::mul_inplace, py::is_operator());

    klass.def(
        "__imul__", [](FreeTensor &self, scalar_t arg) {
            return self.smul_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__imul__", [](FreeTensor &self, long long arg) {
            return self.smul_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](FreeTensor &self, scalar_t arg) {
            return self.sdiv_inplace(scalars::Scalar(arg));
        },
        py::is_operator());
    klass.def(
        "__itruediv__", [](FreeTensor &self, long long arg) {
            return self.sdiv_inplace(scalars::Scalar(self.coeff_type(), arg, 1LL));
        },
        py::is_operator());

    klass.def("add_scal_mul", &FreeTensor::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &FreeTensor::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &FreeTensor::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &FreeTensor::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &FreeTensor::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &FreeTensor::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &FreeTensor::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &FreeTensor::mul_sdiv, "other"_a, "scalar"_a);

    klass.def("exp", &FreeTensor::exp);
    klass.def("log", &FreeTensor::log);
    klass.def("inverse", &FreeTensor::inverse);
    klass.def("fmexp", &FreeTensor::fmexp, "other"_a);

    klass.def("__str__", [](const FreeTensor &self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
    });

    klass.def("__repr__", [](const FreeTensor &self) {
        std::stringstream ss;
        ss << "FreeTensor(width=" << self.width().value()
           << ", depth=" << self.depth().value();
        ss << ", ctype=" << self.coeff_type()->info().name << ')';
        return ss.str();
    });

    klass.def("__eq__", [](const FreeTensor &lhs, const FreeTensor &rhs) { return lhs == rhs; });
    klass.def("__neq__", [](const FreeTensor &lhs, const FreeTensor &rhs) { return lhs != rhs; });

#ifdef ROUGHPY_WITH_NUMPY
    klass.def("__array__", [](const FreeTensor &self) {
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
