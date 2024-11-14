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


#include "lie.h"

RPY_WARNING_PUSH
RPY_MSVC_DISABLE_WARNING(4661)

#include <pybind11/operators.h>

#include "roughpy/core/check.h"                  // for throw_exception, RPY...
#include "roughpy/core/macros.h"                 // for RPY_MSVC_DISABLE_WAR...
#include "roughpy/core/types.h"                  // for key_type, deg_t

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_type.h>

#include "args/kwargs_to_vector_construction.h"
#include "args/numpy.h"
#include "args/parse_data_argument.h"
#include "scalars/scalars.h"

#include "lie_key.h"
#include "setup_algebra_type.h"


using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

static const char* LIE_DOC = R"edoc(

Lie elements live in the free Lie Algebra.
Group-like elements have a one-to-one correspondence with a :class:`Stream`.
That is, for every group-like element, there exists a :class:`Stream` where the :py:meth:`~signature` of that :class:`Stream` is the group-like element.
For more information on Lie Algebras, see `Reutenauer <https://books.google.co.uk/books?id=cBvvAAAAMAAJ&redir_esc=y>`_ and `Bourbaki <https://link.springer.com/book/9783540642428>`_.

You will most commonly encounter :class:`Lie` objects when taking the :py:meth:`~log_signature` of a path.
We can use the Dynkin map to transfer between :class:`Lie` and :py:attr:`~signature` objects.

To construct a :class:`Lie`, you will need :py:data:`~data`. For example, we can construct a :class:`Lie` using a list of polynomials.

.. code:: python

    >>> lie_data_x = [
        1 * roughpy.Monomial("x1"),  # channel (1)
        1 * roughpy.Monomial("x2"),  # channel (2)
        1 * roughpy.Monomial("x3"),  # channel (3)
        1 * roughpy.Monomial("x4"),  # channel ([1, 2])
        1 * roughpy.Monomial("x5"),  # channel ([1, 3])
        1 * roughpy.Monomial("x6"),  # channel ([2, 3])
        ]

    >>> lie_x = rp.Lie(lie_data_x, width=3, depth=2, dtype=rp.RationalPoly)
    >>> print(f"{lie_x=!s}")
        lie_x={ { 1(x1) }(1) { 1(x2) }(2) { 1(x3) }(3) { 1(x4) }([1,2]) { 1(x5) }([1,3]) { 1(x6) }([2,3]) }

You will also need to provide the following parameters:

:py:attr:`ctx`
    Provide an algebra context in which to create the algebra, takes priority over the next 3.

Or

:py:attr:`dtype`
    Scalar type for the algebra (deprecated, use :py:attr:`ctx` instead). Can be a RoughPy data type (:py:attr:`rp.SPReal`, :py:attr:`rp.DPReal`, :py:attr:`rp.Rational`, :py:attr:`rp.PolyRational`), or a Numpy dtype.

:py:attr:`depth`
    Maximum degree for :class:`Lie` and :class:`FreeTensor` objects, etc. (deprecated, use :py:attr:`ctx` instead)

:py:attr:`width`
    Alphabet size, dimension of the underlying space (deprecated, use :py:attr:`ctx` instead)

Optional parameters:

:py:attr:`vector_type`
    :py:attr:`dense` or :py:attr:`sparse`

:py:attr:`keys`
    List/array of :py:attr:`keys` to go along with scalars provided as an array argument.

)edoc";

static Lie construct_lie(py::object data, py::kwargs kwargs)
{
    auto helper = python::kwargs_to_construction_data(kwargs);

    python::DataArgOptions options;
    options.scalar_type = helper.ctype;
    options.allow_scalar = false;
    options.max_nested = 1;

    auto parsed_data = python::parse_data_argument(data, options);

    // bool is_sparse = false;
    scalars::KeyScalarArray buffer;
    if (parsed_data.size() == 1) {
        auto& leaf = parsed_data.back();
        buffer = std::move(leaf.data);
        // is_sparse = leaf.value_type == python::ValueType::KeyValue;
    }

    if (helper.ctype == nullptr) {
        if (options.scalar_type == nullptr) {
            RPY_THROW(py::value_error,"could not deduce an appropriate scalar_type"
            );
        }
        helper.ctype = options.scalar_type;
    }

    if (helper.width == 0 && buffer.size() > 0) {
        helper.width = static_cast<deg_t>(buffer.size());
    }

    if (!helper.ctx) {
        if (helper.width == 0 || helper.depth == 0) {
            RPY_THROW(py::value_error,
                    "you must provide either context or both width and depth"
            );
        }
        helper.ctx = get_context(helper.width, helper.depth, helper.ctype, {});
    }

    auto result = helper.ctx->construct_lie({std::move(buffer), helper.vtype});


    return result;
}

void python::init_lie(py::module_& m)
{

    py::options options;
    options.disable_function_signatures();

    pybind11::class_<Lie> klass(m, "Lie", LIE_DOC);
    klass.def(py::init(&construct_lie), "data"_a = py::none());

    setup_algebra_type(klass);

    klass.def("__getitem__", [](const Lie& self, key_type key) {
        return self[key];
    });
    klass.def("__getitem__", [](const Lie& self, const PyLieKey& lkey) {
             return self[static_cast<key_type>(lkey)];
         });

    klass.def("__repr__", [](const Lie& self) {
        std::stringstream ss;
        ss << "Lie(width=" << *self.width() << ", depth=" << *self.depth();
        ss << ", ctype=" << self.coeff_type()->name() << ')';
//        self->print(ss);
        return ss.str();
    });
}


RPY_WARNING_POP
