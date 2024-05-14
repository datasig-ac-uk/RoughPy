// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "free_tensor.h"

RPY_WARNING_PUSH
RPY_MSVC_DISABLE_WARNING(4661)

#include <sstream>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/scalars/scalar.h>

#include "args/kwargs_to_vector_construction.h"
#include "args/numpy.h"
#include "args/parse_data_argument.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"

#include "setup_algebra_type.h"
#include "tensor_key.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char* FREE_TENSOR_DOC
        = R"eadoc(Element of the (truncated) tensor algebra.

A :class:`FreeTensor` object supports arithmetic operators, providing both objects are compatible,
along with comparison operators. The multiplication operator for this class is the free tensor
multiplication (concatenation of tensor words). Moreover, :class:`FreeTensor` objects are
`iterable <https://docs.python.org/3/glossary.html#term-iterable>`_, where the items are tuples
of :class:`TensorKey` and :class:`float` corresponding to the non-zero elements of the
:class:`FreeTensor`.

The class also supports (implicit and explicit) conversion to a Numpy array type, so it can be
used as an argument to any function that takes Numpy arrays. The array representation of a
:class:`FreeTensor` is one-dimensional. Alternatively, one can construct d-dimensional arrays
containing the elements of degree d by using the :py:meth:`degree_array` method.

There are methods for computing the tensor exponential, :py:meth:`exp`,
logarithm, :py:meth:`log`, and the antipode, :py:meth:`antipode`. See the
documentation of these methods for more information.

A tensor can be created from an array-like object containing the coefficients of the keys, in
their standard order. Since tensors must be created with both an alphabet size and depth, we need
to provide at least the :py:attr:`~depth` argument. However, it is recommended that you also provided
the :py:attr:`~width` argument, otherwise it is assumed that the tensor has degree 1 and the alphabet
size will be determined from the length of the argument.

.. code:: python

    >>> ts1 = rp.FreeTensor([1.0, 2.0, 3.0], depth=2)
    >>> print(ts1)
    { 1() 2(1) 3(2) }
    >>> ts2 = rp.FreeTensor([1.0, 2.0, 3.0], width=2, depth=2)
    >>> print(ts2)
    { 1() 2(1) 3(2) }

If the width argument is provided, this construction can be used to construct :class:`FreeTensor` objects of any
degree, up to the maximum. The :class:`Context` class provides a method
:py:meth:`tensor_size` that can be used to get the dimension of the
tensor up to a given degree.
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

    python::DataArgOptions options;
    options.scalar_type = helper.ctype;
    options.alternative_key = &alt;
    options.max_nested = 1;

    auto parsed_data = python::parse_data_argument(data, options);

    bool is_sparse = false;
    scalars::KeyScalarArray buffer;
    if (parsed_data.size() == 1) {
        auto& leaf = parsed_data.back();
        buffer = std::move(leaf.data);
        is_sparse = leaf.value_type == python::ValueType::KeyValue;
    } else {
    }

    if (helper.ctype == nullptr) {
        if (options.scalar_type == nullptr) {
            RPY_THROW(py::value_error, "could not deduce appropriate scalar type");
        }
        helper.ctype = options.scalar_type;
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
        if (is_sparse) {
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
    klass.def("__getitem__", [](const FreeTensor& self, const PyTensorKey& tkey) {
             return self[static_cast<key_type>(tkey)];
         });

    klass.def("exp", &FreeTensor::exp, "Computes the truncated exponential of a :class:`~FreeTensor` instance.");
    klass.def("log", &FreeTensor::log, "Computes the truncated log of the argument up to degree :py:attr:`~max_degree`");
    klass.def("antipode", &FreeTensor::antipode, "Compute the antipode of a :class:`~FreeTensor` instance");
//    klass.def("inverse", &FreeTensor::inverse);
    klass.def("fmexp", &FreeTensor::fmexp, "other"_a, "Fused multiply exponential operation for :py:attr:`~FreeTensor` objects. Computes :math:`a exp(x)`.");
//
    klass.def("__repr__", [](const FreeTensor& self) {
        std::stringstream ss;
        ss << "FreeTensor(width=" << *self.width()
           << ", depth=" << *self.depth();
        ss << ", ctype=" << self.coeff_type()->name() << ')';
//        self->print(ss);
        return ss.str();
    });
}


RPY_WARNING_POP
