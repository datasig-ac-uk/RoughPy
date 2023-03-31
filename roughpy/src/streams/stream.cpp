// Copyright (c) 2023 Datasig Group. All rights reserved.
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

#include "stream.h"

#include <cmath>
#include <cstring>
#include <sstream>

#include <roughpy/streams/stream.h>
#include <roughpy/scalars/scalar_type.h>

#include "algebra/context.h"
#include "intervals/interval.h"
#include "scalars/scalar_type.h"
#include "args/numpy.h"

#ifdef ROUGHPY_WITH_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>
#endif

using namespace rpy;
using namespace rpy::streams;

static const char *STREAM_DOC = R"rpydoc(
A stream is an abstract stream of data viewed as a rough path.
)rpydoc";


struct SigArgs {
    optional<intervals::RealInterval> interval;
    resolution_t resolution;
    algebra::context_pointer ctx;
};

static int resolution_converter(PyObject* object, void* out) {
    if (Py_TYPE(object) == &PyFloat_Type) {
        auto tmp = PyFloat_AsDouble(object);
        int exponent;
        frexp(tmp, &exponent);
        *reinterpret_cast<resolution_t*>(out) = -std::min(0, exponent-1);
    } else if (Py_TYPE(object) == &PyLong_Type) {
        *reinterpret_cast<resolution_t*>(out) = PyLong_AsLong(object);
    } else if (Py_IsNone(object)) {
        // do nothing, use default
    } else {
        PyErr_SetString(PyExc_TypeError, "resolution should be either float or int");
        return 0;
    }
    return 1;
}

static int parse_sig_args(PyObject* args, PyObject* kwargs, const StreamMetadata* smeta, SigArgs* sigargs) {
    static const char *kwords[] = {
        "interval_or_inf",
        "sup",
        "resolution",
        "ctx",
        "depth",
        "dtype",
        nullptr};

    PyObject *interval_or_inf;
    double sup = std::nan("ns");
    resolution_t resolution = smeta->default_resolution;
    PyObject *ctx = nullptr;
    int depth = 0;
    PyObject *dtype = nullptr;


    auto result = PyArg_ParseTupleAndKeywords(args, kwargs,
                                              "|OdO&$O!iO",
                                              const_cast<char **>(kwords),
                                              &interval_or_inf,
                                              &sup,
                                              &resolution_converter,
                                              &resolution,
                                              &python::RPyContext_Type,
                                              &ctx,
                                              &depth,
                                              &dtype);
    if (result == 0) {
        return -1;
    }

    sigargs->resolution = resolution;

    // First decide if we're given an interval, inf/sup pair, or global
    if (interval_or_inf == nullptr) {
        // Global, nothing to do as optional default is empty.
    }
    else if (Py_TYPE(interval_or_inf) == &PyFloat_Type || Py_TYPE(interval_or_inf) == &PyLong_Type) {
        param_t inf;
        if (Py_TYPE(interval_or_inf) == &PyFloat_Type) {
            inf = PyFloat_AsDouble(interval_or_inf);
        } else {
            inf = PyLong_AsDouble(interval_or_inf);
        }

        if (std::isnan(sup)) {
            PyErr_SetString(PyExc_ValueError, "inf and sup must both be specified");
            return -1;
        }

        if (inf > sup) {
            PyErr_SetString(PyExc_ValueError, "inf must not be larger than sup");
            return -1;
        }

        sigargs->interval = intervals::RealInterval(inf, sup);
    } else if (Py_TYPE(interval_or_inf) == (PyTypeObject*) py::type::of<intervals::Interval>().ptr()) {
        sigargs->interval = intervals::RealInterval(py::handle(interval_or_inf).cast<const intervals::Interval&>());
    } else if (Py_TYPE(interval_or_inf) == (PyTypeObject*) py::type::of<intervals::RealInterval>().ptr()) {
        sigargs->interval = py::handle(interval_or_inf).cast<const intervals::RealInterval&>();
    } else if (Py_TYPE(interval_or_inf) == (PyTypeObject*) py::type::of<intervals::DyadicInterval>().ptr()) {
        sigargs->interval = intervals::RealInterval(py::handle(interval_or_inf).cast<const intervals::DyadicInterval&>());
    } else {
        PyErr_SetString(PyExc_TypeError, "unexpected type for argument interval_or_inf");
        return -1;
    }

    // Finally we need to decide if a change of context is necessary.
    if (ctx == nullptr) {
        if (depth == 0 && dtype == nullptr) {
            sigargs->ctx = smeta->default_context;
            return 0;
        }

        const scalars::ScalarType *ctype = smeta->data_scalar_type;
        if (dtype != nullptr) {
            if (Py_TYPE(dtype) == (PyTypeObject *)python::get_scalar_metaclass().ptr()) {
                ctype = ((python::PyScalarMetaType *)dtype)->tp_ctype;
            } else if (Py_TYPE(dtype) == &PyUnicode_Type) {
                const auto *dtype_str = PyUnicode_AsUTF8(dtype);
                ctype = scalars::get_type(dtype_str);
                if (ctype == nullptr) {
                    PyErr_SetString(PyExc_TypeError, "unrecognised scalar type id");
                    return -1;
                }
            }
#ifdef ROUGHPY_WITH_NUMPY
            else if (Py_TYPE(dtype) == &PyArrayDescr_Type) {
                ctype = python::npy_dtype_to_ctype(py::reinterpret_borrow<py::dtype>(dtype));
                if (ctype == nullptr) {
                    PyErr_SetString(PyExc_ValueError, "unsupported scalar type");
                    return -1;
                }
            }
#endif
            else {
                PyErr_SetString(PyExc_TypeError, "unexpected argument type for argument dtype");
                return -1;
            }
        }

        sigargs->ctx = smeta->default_context->get_alike(depth, ctype);
    } else {
        // a provided context always takes priority over other configurations.
        // This cast is infallible because we've already checked that the type is correct.
        sigargs->ctx = python::ctx_cast(ctx);
    }

    return 0;
}


extern "C" {

static const char *SIGNATURE_DOC = R"rpydoc(Compute the signature of the stream over an interval.)rpydoc";
static PyObject *signature(PyObject *self, PyObject *args, PyObject *kwargs) {
    SigArgs sigargs;
    auto* stream = (python::RPyStream*) self;

    if (parse_sig_args(args, kwargs, &stream->m_data.metadata(), &sigargs) < 0) {
        return nullptr;
    }

    algebra::FreeTensor result;
    try {
        if (sigargs.interval) {
            result = stream->m_data.signature(sigargs.interval.value(), sigargs.resolution, *sigargs.ctx);
        } else {
            result = stream->m_data.signature(sigargs.resolution, *sigargs.ctx);
        }
    } catch (std::exception& err) {
        PyErr_SetString(PyExc_RuntimeError, err.what());
        return nullptr;
    }

    return py::cast(std::move(result)).release().ptr();
}

static const char *LOGSIGNATURE_DOC = R"rpydoc(Compute the log signature of the stream over an interval.)rpydoc";
static PyObject *log_signature(PyObject *self, PyObject *args, PyObject *kwargs) {
    SigArgs sigargs;
    auto *stream = (python::RPyStream *)self;

    if (parse_sig_args(args, kwargs, &stream->m_data.metadata(), &sigargs) < 0) {
        return nullptr;
    }

    algebra::Lie result;
    if (sigargs.interval) {
        result = stream->m_data.log_signature(sigargs.resolution, *sigargs.ctx);
    } else {
        result = stream->m_data.log_signature(sigargs.interval.value(), sigargs.resolution, *sigargs.ctx);
    }

    return py::cast(result).release().ptr();
}

static const char* SIG_DERIV_DOC = R"rpydoc(Compute the derivative of a signature calculation with respect
to a perturbation of the underlying path.
)rpydoc";
static PyObject* sig_deriv(PyObject* self, PyObject* args, PyObject* kwargs) {
    Py_RETURN_NONE;
}


static PyMethodDef RPyStream_members[] = {
    {"signature", (PyCFunction)&signature, METH_VARARGS | METH_KEYWORDS, SIGNATURE_DOC},
    {"log_signature", (PyCFunction)&log_signature, METH_VARARGS | METH_KEYWORDS, LOGSIGNATURE_DOC},
    {"signature_derivative", (PyCFunction) &sig_deriv, METH_VARARGS | METH_KEYWORDS, SIG_DERIV_DOC},
    {nullptr, nullptr, 0, nullptr}};

static PyObject *width_getter(PyObject *self) {
    return PyLong_FromUnsignedLong(reinterpret_cast<python::RPyStream*>(self)->m_data.metadata().width);
}

static PyGetSetDef RPyStream_getset[] = {
    {"width", (getter)width_getter, nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

static PyObject *RPyStream_repr(PyObject *self) {
    std::stringstream ss;
    ss << "Stream(width=" << reinterpret_cast<python::RPyStream*>(self)->m_data.metadata().width << ')';
    return PyUnicode_FromString(ss.str().c_str());
}
static PyObject *RPyStream_str(PyObject *self) {
    return RPyStream_repr(self);
}

PyTypeObject rpy::python::RPyStream_Type =
    {
        PyVarObject_HEAD_INIT(nullptr, 0) "_roughpy.Stream", /* tp_name */
        sizeof(python::RPyStream),                     /* tp_basicsize */
        0,                                             /* tp_itemsize */
        nullptr,                                       /* tp_dealloc */
        0,                                             /* tp_vectorcall_offset */
        (getattrfunc) nullptr,                         /* tp_getattr */
        (setattrfunc) nullptr,                         /* tp_setattr */
        nullptr,                                       /* tp_as_async */
        (reprfunc)RPyStream_repr,                      /* tp_repr */
        nullptr,                                       /* tp_as_number */
        nullptr,                                       /* tp_as_sequence */
        nullptr,                                       /* tp_as_mapping */
        nullptr,                   /* tp_hash */
        nullptr,                                       /* tp_call */
        (reprfunc) RPyStream_str,                                   /* tp_str */
        nullptr,                       /* tp_getattro */
        (setattrofunc) nullptr,                        /* tp_setattro */
        (PyBufferProcs *)nullptr,                            /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,      /* tp_flags */
        STREAM_DOC,                                    /* tp_doc */
        nullptr,                                       /* tp_traverse */
        nullptr,                                       /* tp_clear */
        nullptr,                                       /* tp_richcompare */
        0,                                             /* tp_weaklistoffset */
        (getiterfunc) nullptr,                         /* tp_iter */
        nullptr,                                       /* tp_iternext */
        RPyStream_members,                             /* tp_methods */
        nullptr,                                       /* tp_members */
        RPyStream_getset,                              /* tp_getset */
        nullptr,                                       /* tp_base */
        nullptr,                                       /* tp_dict */
        nullptr,                                       /* tp_descr_get */
        nullptr,                                       /* tp_descr_set */
        0,                                             /* tp_dictoffset */
        (initproc) nullptr,                            /* tp_init */
        nullptr,                                       /* tp_alloc */
        nullptr,                             /* tp_new */
};
}

PyObject *python::RPyStream_FromStream(Stream &&stream) {
    auto* py_stream = RPyStream_Type.tp_alloc(&RPyStream_Type, 0);
    if (py_stream == nullptr) {
        return nullptr;
    }

    auto* dst = &reinterpret_cast<python::RPyStream*>(py_stream)->m_data;

    ::new (dst) Stream(std::move(stream));

    return reinterpret_cast<PyObject*>(py_stream);
}

void python::init_stream(py::module_ &m) {

    if (PyType_Ready(&RPyStream_Type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("Stream", (PyObject*) &RPyStream_Type);

}
