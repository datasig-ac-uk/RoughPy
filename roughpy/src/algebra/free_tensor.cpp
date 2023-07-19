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

#include "free_tensor.h"

#include <sstream>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/scalars/scalar.h>

#include "args/kwargs_to_vector_construction.h"
#include "args/numpy.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"

#include "setup_algebra_type.h"
#include "tensor_key.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char* FREE_TENSOR_DOC
        = R"eadoc(Element of the (truncated) tensor algebra.

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

static FreeTensor construct_free_tensor(py::object data, py::kwargs kwargs)
{
    auto helper = python::kwargs_to_construction_data(kwargs);

    auto py_key_type = py::type::of<python::PyTensorKey>();
    python::AlternativeKeyType alt{
            py_key_type, [](py::handle py_key) -> key_type {
                return static_cast<key_type>(py_key.cast<python::PyTensorKey>()
                );
            }};

    python::PyToBufferOptions options;
    options.type = helper.ctype;
    options.alternative_key = &alt;

    auto buffer = python::py_to_buffer(data, options);

    if (helper.ctype == nullptr) {
        if (options.type == nullptr) {
            RPY_THROW(py::value_error, "could not deduce appropriate scalar type");
        }
        helper.ctype = options.type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size()) - 1;
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            RPY_THROW(py::value_error,
                    "you must provide either context or both width and depth"
            );
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

    auto result = helper.ctx->construct_free_tensor(
            {std::move(buffer), helper.vtype}
    );

    if (options.cleanup) { options.cleanup(); }

    RPY_DBG_ASSERT(result.coeff_type() != nullptr);

    return result;
}

void python::init_free_tensor(py::module_& m)
{

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<FreeTensor> klass(m, "FreeTensor", FREE_TENSOR_DOC);
    klass.def(py::init(&construct_free_tensor), "data"_a = py::none());

    python::setup_algebra_type(klass);

    klass.def("__getitem__", [](const FreeTensor& self, key_type key) {
        return self[key];
    });

    klass.def("exp", &FreeTensor::exp);
    klass.def("log", &FreeTensor::log);
    klass.def("antipode", &FreeTensor::antipode);
//    klass.def("inverse", &FreeTensor::inverse);
    klass.def("fmexp", &FreeTensor::fmexp, "other"_a);
//
    klass.def("__repr__", [](const FreeTensor& self) {
        std::stringstream ss;
        ss << "FreeTensor(width=" << *self.width()
           << ", depth=" << *self.depth();
        ss << ", ctype=" << self.coeff_type()->info().name << ')';
//        self->print(ss);
        return ss.str();
    });
}
