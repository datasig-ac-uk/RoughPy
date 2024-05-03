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

#include "shuffle_tensor.h"

RPY_WARNING_PUSH
RPY_MSVC_DISABLE_WARNING(4661)

#include <sstream>

#include <pybind11/operators.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/algebra/context.h>
#include <roughpy/algebra/shuffle_tensor.h>

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

static const char* SHUFFLE_TENSOR_DOC
        = R"eadoc(Element of the shuffle tensor algebra.

:class:`ShuffleTensor` objects are one way of representing the linear functionals on :class:`FreeTensor` objects.
The shuffle product corresponds to point-wise multiplication of the continuous functions on paths via the signature correspondence.
For more information on shuffle tensors, see
`Reutenauer, Free Lie Algebras <https://www.sciencedirect.com/science/article/abs/pii/S157079540380075X>`_.

Shuffle tensors are useful because they represent functions on paths via the :py:meth:`signature`.

You can construct :class:`ShuffleTensor` objects in the following way, here we use polynomial coefficients:

.. code:: python

    >>> shuffle_tensor = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2, depth=2, dtype=roughpy.RationalPoly)

Which would look like this:

.. code:: python

    { { 1(x0) }() { 1(x1) }(1) { 1(x2) }(2) { 1(x3) }(1,1) { 1(x4) }(1,2) { 1(x5) }(2,1) { 1(x6) }(2,2) }

You construct with data, which for the example above was the following list:

.. code:: python

    [{ 1(x0) }, { 1(x1) }, { 1(x2) }, { 1(x3) }, { 1(x4) }, { 1(x5) }, { 1(x6) }]

As well as data, you will need to provide the following parameters:

:py:attr:`ctx`
    Provide an algebra context in which to create the algebra, takes priority over the next 3.

Or

:py:attr:`dtype`
    Scalar type for the algebra (deprecated, use :py:attr:`ctx` instead). Can be a RoughPy data type (:py:attr:`rp.SPReal`, :py:attr:`rp.DPReal`, :py:attr:`rp.Rational`, :py:attr:`rp.PolyRational`), or a Numpy dtype.

:py:attr:`depth`
    Maximum degree for :class:`Lie` objects, :class:`FreeTensor` objects, etc. (deprecated, use :py:attr:`ctx` instead)

:py:attr:`width`
    Alphabet size, dimension of the underlying space (deprecated, use :py:attr:`ctx` instead)

Optional parameters:

:py:attr:`vector_type`
    :py:attr:`dense` or :py:attr:`sparse`

:py:attr:`keys`
    List/array of :py:attr:`keys` to go along with scalars provided as an array argument.

You can shuffle two tensors together, using ``*``. For example, using ``x`` and ``y`` for indeterminate names for ``shuffle_tensor1`` and ``shuffle_tensor2``, for the above tensor we could do:

.. code:: python

    >>> result = shuffle_tensor1*shuffle_tensor2

The result would look like this:

.. code:: python

    result={ { 1(x0 y0) }() { 1(x0 y1) 1(x1 y0) }(1) { 1(x0 y2) 1(x2 y0) }(2) { 1(x0 y3) 2(x1 y1) 1(x3 y0) }(1,1) { 1(x0 y4) 1(x1 y2) 1(x2 y1) 1(x4 y0) }(1,2) { 1(x0 y5) 1(x1 y2) 1(x2 y1) 1(x5 y0) }(2,1) { 1(x0 y6) 2(x2 y2) 1(x6 y0) }(2,2) }


)eadoc";

static ShuffleTensor construct_shuffle(py::object data, py::kwargs kwargs)
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
        if (buffer.has_keys()) {
            // if data comes and k-v pairs, then it is reasonable to assume
            // the user wants a sparse tensor.
            helper.vtype = VectorType::Sparse;
        } else {
            // otherwise dense
            helper.vtype = VectorType::Dense;
        }
    }

    auto result = helper.ctx->construct_shuffle_tensor(
            {std::move(buffer), helper.vtype}
    );


    return result;
}

void rpy::python::init_shuffle_tensor(py::module_& m)
{
    py::options options;
    options.disable_function_signatures();

    py::class_<ShuffleTensor> klass(m, "ShuffleTensor", SHUFFLE_TENSOR_DOC);
    klass.def(py::init(&construct_shuffle), "data"_a = py::none());
    setup_algebra_type(klass);

    klass.def("__getitem__", [](const ShuffleTensor& self, key_type key) {
        return self[key];
    });
    klass.def("__getitem__", [](const ShuffleTensor& self, const PyTensorKey& tkey) {
             return self[static_cast<key_type>(tkey)];
         });


    klass.def(
            "__matmul__",
            [](const ShuffleTensor& shuf, const FreeTensor& arg) {
                scalars::Scalar result(shuf.coeff_type(), 0, 1);
                for (auto&& item : shuf) {
                    result += item.value() * arg[item.key()];
                }
                return result;
            },
            py::is_operator()
    );

    klass.def("__repr__", [](const ShuffleTensor& self) {
        std::stringstream ss;
        ss << "ShuffleTensor(width=" << *self.width()
           << ", depth=" << *self.depth();
        ss << ", ctype=" << self.coeff_type()->name() << ')';
        return ss.str();
    });
}


RPY_WARNING_POP
