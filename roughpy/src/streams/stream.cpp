// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

#include "stream.h"

#include <cmath>
#include <cstring>
#include <sstream>

#include <roughpy/intervals/partition.h>
#include <roughpy/platform/archives.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/streams/stream.h>

#include "algebra/context.h"
#include "args/kwargs_to_path_metadata.h"
#include "args/numpy.h"
#include "intervals/interval.h"
#include "scalars/scalar_type.h"

#ifdef ROUGHPY_WITH_NUMPY
#  define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION

#  include <numpy/arrayobject.h>

#endif

using namespace rpy;
using namespace rpy::streams;

static const char* STREAM_DOC = R"rpydoc(

A :class:`Stream` means an object that provides the :py:meth:`~signature` or :py:meth:`~log_signature` over any :class:`Interval`.
For more information on :class:`Stream` objects, see `Lyons and McLeod <http://arxiv.org/abs/2206.14674>`_ and `Lyons et al. <https://link.springer.com/book/10.1007/978-3-540-71285-5>`_.

:class:`Stream` objects are parametrised sequential data viewed via Rough Path theory as a rough path.

You can construct a :class:`Stream` in many ways. You can use Lie increments:

.. code:: python

    >>> roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

Tick streams:

.. code:: python

    # create tick stream data
    >>> data = { 1.0: [("first", "increment", 1.0),("second", "increment", 2.0)],
     2.0: [("first", "increment", 1.0)]
     }

    # construct stream
    >>> tick_stream = TickStream.from_data(data, width=2, depth=2, dtype=DPReal)

Brownian streams:

.. code:: python

    # Generating on demand from a source of randomness with normal increments that approximate Brownian motion
    >>> brownian_stream = BrownianStream.with_generator(width=2, depth=2, dtype=DPReal)

Piecewise abelian streams:

.. code:: python

    # create piecewise lie data
    >>> piecewise_intervals = [RealInterval(float(i), float(i + 1)) for i in range(5)]

    >>> piecewise_lie_data = [
    (interval,
    brownian_stream.log_signature(interval))
    for interval in piecewise_intervals
    ]

    # construct stream
    >>> piecewise_abelian_stream = PiecewiseAbelianStream.construct(piecewise_lie_data, width=2, depth=2, dtype=DPReal)

Function streams:

.. code:: python

    # create a function to generate a stream from
    >>> def func(t, ctx):
    ...     return Lie(np.array([t, 2*t]), ctx=ctx)

    #construct stream
    >>> function_stream = rp.FunctionStream.from_function(func, width=2, depth=2, dtype=rp.DPReal)

External source data:

.. code:: python

    # create a stream from an external source
    # here we use a sound file, but other formats are supported
    >>> roughpy.ExternalDataStream.from_uri("/path/to/sound_file.mp3", depth=2)

All of these :class:`Stream` objects are constructed with different data types.

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

:class:`Stream` objects also have the following optional parameters:

:py:attr:`schema`
    An abstract description of what the comprises the channels of the underlying space in schema form. Can be deduced from data/constructor. If provided must be "correct" (i.e. :py:attr:`width` must be correct).

:py:attr:`channel_types`
    Sequence of channel types, ``str`` name of channel type (e.g. ``increment``), or ``dict`` of ``name: type`` pairs. Used to construct a schema if you don't have one of those already.

:py:attr:`include_time`
    Bool, indicates whether the parameter value should be included as a :class:`stream` channel.

:py:attr:`vtype`
    Default vector type for algebras return from :class:`Stream` methods. (:py:attr:`dense` or :py:attr:`sparse`). Default is currently :py:attr:`dense`, although this will change to be determined by the form of the underlying :class:`Stream`.

:py:attr:`resolution`
    Resolution for the dyadic dissection of the domain, and the default :py:attr:`resolution` used in :py:meth:`~signature` / :py:meth:`~log_signature` calculations.

:py:attr:`support`
    :class:`Interval` of parameter values on which the :class:`Stream` has meaning.

:py:attr:`indices`
    For :py:attr:`~LieIncrementStreams`, optionally provide a list/array of parameter values at which each row of the input data occurs. (default, row "i" occurs at parameter value "i"), or integer indicating the "column" of data that corresponds to the parameter. (Must be present in all rows)


)rpydoc";

struct SigArgs {
    optional<intervals::RealInterval> interval;
    optional<resolution_t> resolution;
    algebra::context_pointer ctx;
};

static int resolution_converter(PyObject* object, void* out)
{
    if (Py_TYPE(object) == &PyFloat_Type) {
        *reinterpret_cast<optional<resolution_t>*>(out)
                = python::param_to_resolution(PyFloat_AsDouble(object));
    } else if (Py_TYPE(object) == &PyLong_Type) {
        *reinterpret_cast<optional<resolution_t>*>(out) = PyLong_AsLong(object);
#if PY_VERSION_HEX >= 0x030A0000
    } else if (Py_IsNone(object)) {
#else
    } else if (object == Py_None) {
#endif
        // do nothing, use default
    } else {
        PyErr_SetString(
                PyExc_TypeError,
                "resolution should be either float or int"
        );
        return 0;
    }
    return 1;
}

static int parse_sig_args(
        PyObject* args,
        PyObject* kwargs,
        const StreamMetadata* smeta,
        SigArgs* sigargs
)
{
    static const char* kwords[]
            = {"interval_or_inf",
               "sup",
               "resolution",
               "ctx",
               "depth",
               "dtype",
               nullptr};

    PyObject* interval_or_inf = nullptr;
    PyObject* ctx = nullptr;
    PyObject* py_sup = nullptr;
    PyObject* dtype = nullptr;
    deg_t depth = 0;

    auto result = PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|OOO&$O!iO",
            const_cast<char**>(kwords),
            &interval_or_inf,
            &py_sup,
            &resolution_converter,
            &sigargs->resolution,
            &python::RPyContext_Type,
            &ctx,
            &depth,
            &dtype
    );
    if (result == 0) { return -1; }

    // First decide if we're given an interval, inf/sup pair, or global
    if (interval_or_inf == nullptr || interval_or_inf == Py_None) {
        // Global, nothing to do as optional default is empty.
    } else if (Py_TYPE(interval_or_inf) == &PyFloat_Type || Py_TYPE(interval_or_inf) == &PyLong_Type) {
        if (py_sup == nullptr) {
            // In this case, we're expecting the first argument to be resolution
            if (Py_TYPE(interval_or_inf) == &PyFloat_Type) {
                PyErr_SetString(
                        PyExc_TypeError,
                        "expecting integer resolution"
                );
                return -1;
            }
            sigargs->resolution
                    = static_cast<resolution_t>(PyLong_AsLong(interval_or_inf));

            sigargs->interval = {};
        } else {
            param_t inf;
            param_t sup;

            if (Py_TYPE(interval_or_inf) == &PyFloat_Type) {
                inf = PyFloat_AsDouble(interval_or_inf);
            } else {
                inf = PyLong_AsDouble(interval_or_inf);
            }

            if (Py_TYPE(py_sup) == &PyFloat_Type) {
                sup = PyFloat_AsDouble(py_sup);
            } else if (Py_TYPE(py_sup) == &PyLong_Type) {
                sup = PyLong_AsDouble(py_sup);
            } else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "expected float value for sup"
                );
                return -1;
            }

            if (inf > sup) {
                PyErr_SetString(
                        PyExc_ValueError,
                        "inf must not be larger than sup"
                );
                return -1;
            }
            sigargs->interval = intervals::RealInterval(inf, sup);
        }
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::RealInterval>()
                          .ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = py::handle(interval_or_inf)
                                    .cast<const intervals::RealInterval&>();
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::DyadicInterval>()
                          .ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = intervals::RealInterval(
                py::handle(interval_or_inf)
                        .cast<const intervals::DyadicInterval&>()
        );
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::Interval>().ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = intervals::RealInterval(
                py::handle(interval_or_inf).cast<const intervals::Interval&>()
        );
    } else {
        PyErr_SetString(
                PyExc_TypeError,
                "unexpected type for argument interval_or_inf"
        );
        return -1;
    }

    // Finally we need to decide if a change of context is necessary.
    if (ctx == nullptr) {
        if (depth == 0 && dtype == nullptr) {
            sigargs->ctx = smeta->default_context;
            return 0;
        }

        const scalars::ScalarType* ctype = smeta->data_scalar_type;
        if (dtype != nullptr) {
            if (Py_TYPE(dtype)
                == (PyTypeObject*) python::get_scalar_metaclass().ptr()) {
                ctype = ((python::PyScalarMetaType*) dtype)->tp_ctype;
            } else if (Py_TYPE(dtype) == &PyUnicode_Type) {
                const auto* dtype_str = PyUnicode_AsUTF8(dtype);
                const auto ctype_o = scalars::get_type(dtype_str);
                if (ctype_o) {
                    PyErr_SetString(
                            PyExc_TypeError,
                            "unrecognised scalar type id"
                    );
                    return -1;
                }
                ctype = *ctype_o;
            }
#ifdef ROUGHPY_WITH_NUMPY
            else if (Py_TYPE(dtype) == &PyArrayDescr_Type) {
                ctype = python::npy_dtype_to_ctype(
                        py::reinterpret_borrow<py::dtype>(dtype)
                );
                if (ctype == nullptr) {
                    PyErr_SetString(
                            PyExc_ValueError,
                            "unsupported scalar type"
                    );
                    return -1;
                }
            }
#endif
            else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "unexpected argument type for argument dtype"
                );
                return -1;
            }
        }

        sigargs->ctx = smeta->default_context->get_alike(depth, ctype);
    } else {
        // a provided context always takes priority over other configurations.
        // This cast is infallible because we've already checked that the type
        // is correct.
        sigargs->ctx = python::ctx_cast(ctx);
    }

    return 0;
}

extern "C" {

static const char* SIGNATURE_DOC
        = R"rpydoc(Compute the :py:meth:`signature` of the :class:`Stream` over an :class:`Interval`.)rpydoc";
static PyObject* signature(PyObject* self, PyObject* args, PyObject* kwargs)
{
    SigArgs sigargs;
    auto* stream = (python::RPyStream*) self;

    if (parse_sig_args(args, kwargs, &stream->m_data.metadata(), &sigargs)
        < 0) {
        return nullptr;
    }

    algebra::FreeTensor result;
    try {
        py::gil_scoped_release gil;
        if (sigargs.interval) {
            if (sigargs.resolution) {
                result = stream->m_data.signature(
                        *sigargs.interval,
                        *sigargs.resolution,
                        *sigargs.ctx
                );
            } else {
                result = stream->m_data.signature(
                        *sigargs.interval,
                        *sigargs.ctx
                );
            }
        } else {
            if (sigargs.resolution) {
                result = stream->m_data.signature(
                        *sigargs.resolution,
                        *sigargs.ctx
                );
            } else {
                result = stream->m_data.signature(*sigargs.ctx);
            }
        }
    } catch (std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        return nullptr;
    }

    return py::cast(std::move(result)).release().ptr();
}

static const char* LOGSIGNATURE_DOC
        = R"rpydoc(Compute the log signature of the :class:`Stream` over an :class:`Interval`.)rpydoc";
static PyObject* log_signature(PyObject* self, PyObject* args, PyObject* kwargs)
{
    SigArgs sigargs;
    auto* stream = (python::RPyStream*) self;

    if (parse_sig_args(args, kwargs, &stream->m_data.metadata(), &sigargs)
        < 0) {
        return nullptr;
    }

    algebra::Lie result;
    try {
        py::gil_scoped_release gil;
        if (sigargs.interval) {
            if (sigargs.resolution) {
                result = stream->m_data.log_signature(
                        *sigargs.interval,
                        *sigargs.resolution,
                        *sigargs.ctx
                );
            } else {
                result = stream->m_data.log_signature(
                        *sigargs.interval,
                        *sigargs.ctx
                );
            }
        } else {
            if (sigargs.resolution) {
                result = stream->m_data.log_signature(
                        *sigargs.resolution,
                        *sigargs.ctx
                );
            } else {
                result = stream->m_data.log_signature(*sigargs.ctx);
            }
        }
    } catch (std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        return nullptr;
    }

    return py::cast(result).release().ptr();
}

static const char* SIG_DERIV_DOC
        = R"rpydoc(Compute the derivative of a signature calculation with respect
to a perturbation of the underlying path.
)rpydoc";
static PyObject* sig_deriv(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[]
            = {"interval", "perturbation", "resolution", "depth", nullptr};

    const auto& stream = reinterpret_cast<python::RPyStream*>(self)->m_data;

    PyObject* arg;
    PyObject* second_arg = nullptr;
    rpy::resolution_t resolution = stream.metadata().default_resolution;
    deg_t depth = -1;
    auto ctx = stream.metadata().default_context;

    auto parse_result = PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "O|Oi$i",
            const_cast<char**>(kwords),
            &arg,
            &second_arg,
            &resolution,
            &depth
    );
    if (!static_cast<bool>(parse_result)) { return nullptr; }

    if (depth != -1) { ctx = ctx->get_alike(depth); }

    auto* interval_type = reinterpret_cast<PyTypeObject*>(
            py::type::of<intervals::Interval>().ptr()
    );
    auto* lie_type
            = reinterpret_cast<PyTypeObject*>(py::type::of<algebra::Lie>().ptr()
            );

    if (second_arg != nullptr && Py_TYPE(second_arg) == &PyLong_Type) {
        resolution = static_cast<resolution_t>(PyLong_AsLong(second_arg));
        second_arg = nullptr;
    }

    algebra::FreeTensor result;
    if (PyObject_IsInstance(arg, (PyObject*) interval_type)
        && second_arg != nullptr && Py_TYPE(second_arg) == lie_type) {

        const auto& interval
                = py::handle(arg).cast<const intervals::Interval&>();
        const auto& perturbation
                = py::handle(second_arg).cast<const algebra::Lie&>();

        result = stream.signature_derivative(
                interval,
                perturbation,
                resolution,
                *ctx
        );

    } else if (PyTuple_Check(arg) && PySequence_Length(arg) == 2) {
        py::handle py_interval;
        py::handle py_perturbation;

        parse_result = PyArg_ParseTuple(
                arg,
                "O!O!",
                interval_type,
                &py_interval.ptr(),
                lie_type,
                &py_perturbation.ptr()
        );
        if (!static_cast<bool>(parse_result)) { return nullptr; }

        const auto& interval = py_interval.cast<const intervals::Interval&>();
        const auto& perturbation = py_perturbation.cast<const algebra::Lie&>();

        result = stream.signature_derivative(
                interval,
                perturbation,
                resolution,
                *ctx
        );
    } else if (static_cast<bool>(PySequence_Check(arg)) && PySequence_Length(arg) > 0) {
        streams::Stream::perturbation_list_t perturbations;
        const auto n_perturbations = PySequence_Length(arg);
        perturbations.reserve(n_perturbations);

        for (Py_ssize_t i = 0; i < n_perturbations; ++i) {
            PyObject* item = PySequence_ITEM(arg, i);

            if (!static_cast<bool>(PySequence_Check(item))
                || PySequence_Length(item) != 2) {
                PyErr_SetString(
                        PyExc_TypeError,
                        "expected interval/perturbation pair"
                );
                return nullptr;
            }

            py::handle py_interval;
            py::handle py_perturbation;

            parse_result = PyArg_ParseTuple(
                    item,
                    "O!O!",
                    interval_type,
                    &py_interval.ptr(),
                    lie_type,
                    &py_perturbation.ptr()
            );
            if (!static_cast<bool>(parse_result)) { return nullptr; }

            perturbations.emplace_back(
                    intervals::RealInterval(
                            py_interval.cast<const intervals::Interval&>()
                    ),
                    py_perturbation.cast<algebra::Lie&>()->borrow_mut()
            );
        }

        result = stream.signature_derivative(perturbations, resolution, *ctx);
    } else {
        PyErr_SetString(
                PyExc_ValueError,
                "unexpected arguments to signature derivative"
        );
        return nullptr;
    }

    return py::cast(result).release().ptr();
}

static const char SIMPLIFY_STREAM_DOC[] = R"rpydoc(Produce a piecewise
abelian path subordinate to the given :class:`Partition`.
)rpydoc";
static PyObject*
simplify_stream(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[]
            = {"partition", "resolution", "ctx", "depth", nullptr};

    const auto& stream = reinterpret_cast<python::RPyStream*>(self)->m_data;

    const auto& md = stream.metadata();

    deg_t depth = 0;
    resolution_t resolution = md.default_resolution;
    PyObject* py_context = nullptr;
    PyObject* py_partition = nullptr;

    // clang-format off
    auto parse_result = PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!|iO!i", const_cast<char**>(kwords),
            py::type::of<intervals::Partition>().ptr(), &py_partition,
            &resolution,
            &python::RPyContext_Type, &py_context,
            &depth
    );
    // clang-format on
    if (parse_result == 0) { return nullptr; }

    const auto& partition = py::cast<const intervals::Partition&>(py_partition);

    if (py_context != nullptr) {
        auto pctx = reinterpret_cast<python::RPyContext*>(py_context)->p_ctx;

        if (pctx->width() != md.width) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "context width must match "
                    "stream width"
            );
            return nullptr;
        }
        if (pctx->ctype() != md.data_scalar_type) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "context scalar type must match"
                    " stream data type"
            );
            return nullptr;
        }

        return python::RPyStream_FromStream(
                stream.simplify(partition, resolution, *pctx)
        );
    }
    if (depth != 0) {
        auto ctx = md.default_context->get_alike(depth);
        return python::RPyStream_FromStream(
                stream.simplify(partition, resolution, *ctx)
        );
    }

    return python::RPyStream_FromStream(stream.simplify(partition, resolution));
}

static const char RESTRICT_DOC[] = R"rpydoc(Create a new :class:`Stream` with the same
 data but restricted to a given :class:`Interval`.)rpydoc";
static PyObject* restrict(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[] = {"interval", nullptr};

    PyObject* py_interval;

    if (PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "O",
                const_cast<char**>(kwords),
                &py_interval
        )
        == 0) {
        return nullptr;
    }

    intervals::RealInterval ivl(py::cast<const intervals::Interval&>(py_interval
    ));

    const auto& stream = reinterpret_cast<python::RPyStream*>(self)->m_data;

    return python::RPyStream_FromStream(stream.restrict(ivl));
}

static PyObject*
stream___getnewargs_ex__(PyObject* self, PyObject* RPY_UNUSED_VAR)
{
    PyObject* data;

    std::stringstream ss;
    {
        rpy::archives::BinaryOutputArchive oar(ss);
        oar(reinterpret_cast<python::RPyStream*>(self)->m_data);
    }

    auto s = ss.str();
    data = PyByteArray_FromStringAndSize(
            s.data(),
            static_cast<Py_ssize_t>(s.size())
    );

    PyObject* new_args = PyTuple_New(0);
    PyObject* new_kwargs = Py_BuildValue("{sO}", "serialized_stream", data);

    return Py_BuildValue("(OO)", new_args, new_kwargs);
}

static PyMethodDef RPyStream_members[] = {
        {           "signature",
         (PyCFunction) &signature,
         METH_VARARGS | METH_KEYWORDS,
         SIGNATURE_DOC                                      },
        {       "log_signature",
         (PyCFunction) &log_signature,
         METH_VARARGS | METH_KEYWORDS,
         LOGSIGNATURE_DOC                                   },
        {"signature_derivative",
         (PyCFunction) &sig_deriv,
         METH_VARARGS | METH_KEYWORDS,
         SIG_DERIV_DOC                                      },
        {            "simplify",
         (PyCFunction) &simplify_stream,
         METH_VARARGS | METH_KEYWORDS,
         SIMPLIFY_STREAM_DOC                                },
        {            "restrict",
         (PyCFunction)& restrict,
         METH_VARARGS | METH_KEYWORDS,
         RESTRICT_DOC                                       },
        {   "__getnewargs_ex__",
         (PyCFunction) stream___getnewargs_ex__,
         METH_NOARGS, nullptr                               },
        {               nullptr,        nullptr,  0, nullptr}
};

static PyObject* width_getter(PyObject* self)
{
    return PyLong_FromUnsignedLong(
            reinterpret_cast<python::RPyStream*>(self)->m_data.metadata().width
    );
}

static PyObject* ctype_getter(PyObject* self)
{
    return python::to_ctype_type(reinterpret_cast<python::RPyStream*>(self)
                                         ->m_data.metadata()
                                         .data_scalar_type)
            .release()
            .ptr();
}

static PyObject* ctx_getter(PyObject* self)
{
    return python::RPyContext_FromContext(
            reinterpret_cast<python::RPyStream*>(self)
                    ->m_data.metadata()
                    .default_context
    );
}

static PyObject* support_getter(PyObject* self)
{
    return py::cast(reinterpret_cast<python::RPyStream*>(self)->m_data.support()
    )
            .release()
            .ptr();
}

static PyObject* resolution_getter(PyObject* self)
{

    return PyLong_FromLong(reinterpret_cast<python::RPyStream*>(self)
                                   ->m_data.metadata()
                                   .default_resolution);
}

static PyGetSetDef RPyStream_getset[] = {
        {     "width",      (getter) width_getter, nullptr, "Alphabet size, dimension of the underlying space (deprecated, use :py:attr:`ctx` instead).", nullptr},
        {     "dtype",      (getter) ctype_getter, nullptr, "Scalar type for the algebra (deprecated, use :py:attr:`ctx` instead). Can be a RoughPy data type (:py:attr:`rp.SPReal`, :py:attr:`rp.DPReal`, :py:attr:`rp.Rational`, :py:attr:`rp.PolyRational`), or a numpy dtype.", nullptr},
        {       "ctx",        (getter) ctx_getter, nullptr, "Provide an algebra context in which to create the algebra.", nullptr},
        {   "support",    (getter) support_getter, nullptr, ":class:`Interval` of parameter values on which the :class:`Stream` has meaning.", nullptr},
        {"resolution", (getter) resolution_getter, nullptr, ":py:attr:`resolution` for the dyadic dissection of the domain, and the default resolution used in signature/log_signature calculations.", nullptr},
        {     nullptr,                    nullptr, nullptr, nullptr, nullptr}
};

static PyObject* RPyStream_repr(PyObject* self)
{
    std::stringstream ss;
    ss << "Stream(width="
       << reinterpret_cast<python::RPyStream*>(self)->m_data.metadata().width
       << ')';
    return PyUnicode_FromString(ss.str().c_str());
}
static PyObject* RPyStream_str(PyObject* self) { return RPyStream_repr(self); }

static void RPyStream_finalize(PyObject* self)
{
    reinterpret_cast<python::RPyStream*>(self)->m_data.~Stream();
}

static PyObject*
RPyStream_new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs)
{
    using namespace rpy::python;
    static const char* kwords[] = {"serialized_stream", nullptr};

    PyByteArrayObject* pybuff = nullptr;
    RPyStream* self;

    if (!PyArg_ParseTupleAndKeywords(
                args,
                kwargs,
                "|$O!:__new__",
                const_cast<char**>(kwords),
                &PyByteArray_Type,
                &pybuff
        )) {
        return nullptr;
    }

    if (pybuff) {
        // The stream is being initialised with previously serialized data.
        self = reinterpret_cast<RPyStream*>(subtype->tp_alloc(subtype, 0));

        // Ensure that the stream object is initialised in the data slot.
        construct_inplace<streams::Stream>(&self->m_data);
        {
            std::string raw(
                    PyByteArray_AS_STRING(pybuff),
                    PyByteArray_GET_SIZE(pybuff)
            );
            std::stringstream ss(raw);
            rpy::archives::BinaryInputArchive iar(ss);
            iar(self->m_data);
        }

        return reinterpret_cast<PyObject*>(self);
    }

    PyErr_SetString(
            PyExc_ValueError,
            "no construction possible with the provided data"
    );
    return nullptr;
}

PyTypeObject rpy::python::RPyStream_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)         //
        "roughpy.Stream",                         /* tp_name */
        sizeof(python::RPyStream),                /* tp_basicsize */
        0,                                        /* tp_itemsize */
        nullptr,                                  /* tp_dealloc */
        0,                                        /* tp_vectorcall_offset */
        (getattrfunc) nullptr,                    /* tp_getattr */
        (setattrfunc) nullptr,                    /* tp_setattr */
        nullptr,                                  /* tp_as_async */
        (reprfunc) RPyStream_repr,                /* tp_repr */
        nullptr,                                  /* tp_as_number */
        nullptr,                                  /* tp_as_sequence */
        nullptr,                                  /* tp_as_mapping */
        nullptr,                                  /* tp_hash */
        nullptr,                                  /* tp_call */
        (reprfunc) RPyStream_str,                 /* tp_str */
        nullptr,                                  /* tp_getattro */
        (setattrofunc) nullptr,                   /* tp_setattro */
        (PyBufferProcs*) nullptr,                 /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
        STREAM_DOC,                               /* tp_doc */
        nullptr,                                  /* tp_traverse */
        nullptr,                                  /* tp_clear */
        nullptr,                                  /* tp_richcompare */
        0,                                        /* tp_weaklistoffset */
        (getiterfunc) nullptr,                    /* tp_iter */
        nullptr,                                  /* tp_iternext */
        RPyStream_members,                        /* tp_methods */
        nullptr,                                  /* tp_members */
        RPyStream_getset,                         /* tp_getset */
        nullptr,                                  /* tp_base */
        nullptr,                                  /* tp_dict */
        nullptr,                                  /* tp_descr_get */
        nullptr,                                  /* tp_descr_set */
        0,                                        /* tp_dictoffset */
        (initproc) nullptr,                       /* tp_init */
        nullptr,                                  /* tp_alloc */
        (newfunc) RPyStream_new,                  /* tp_new */
        nullptr,                                  /* tp_free */
        nullptr,                                  /* tp_is_gc */
        nullptr,                                  /* tp_bases */
        nullptr,                                  /* tp_mro */
        nullptr,                                  /* tp_cache */
        nullptr,                                  /* tp_subclasses */
        nullptr,                                  /* tp_weaklist */
        nullptr,                                  /* tp_del */
        0,                                        /* tp_version_tag */
        (destructor) RPyStream_finalize,          /* tp_finalize */
        nullptr                                   /* tp_vectorcall */
};
}// extern "C"

PyObject* python::RPyStream_FromStream(Stream&& stream)
{
    auto* py_stream = RPyStream_Type.tp_alloc(&RPyStream_Type, 0);
    if (py_stream == nullptr) { return nullptr; }

    auto* dst = &reinterpret_cast<python::RPyStream*>(py_stream)->m_data;

    ::new (dst) Stream(std::move(stream));

    return reinterpret_cast<PyObject*>(py_stream);
}

void python::init_stream(py::module_& m)
{

    if (PyType_Ready(&RPyStream_Type) < 0) { throw py::error_already_set(); }

    m.add_object("Stream", (PyObject*) &RPyStream_Type);
}
