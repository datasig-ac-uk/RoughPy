//
// Created by sammorley on 10/12/24.
//

#include "tensor_valued_stream.h"

#include <memory>

#include "roughpy/algebra/free_tensor.h"
#include "roughpy/streams/value_stream.h"

#include "stream.h"
#include "signature_arguments.h"

using namespace pybind11::literals;
using namespace rpy;
using namespace rpy::streams;

using algebra::FreeTensor;


struct RPySimpleTensorValuedStream
{
    PyObject_VAR_HEAD//
    std::shared_ptr<const ValueStream<FreeTensor>> p_data;
};


py::object python::TensorValuedStream_FromPtr(
    std::shared_ptr<const ValueStream<FreeTensor>> ptr)
{
    auto new_obj = py::reinterpret_steal<py::object>(
        TensorValuedStream_Type.tp_alloc(&TensorValuedStream_Type, 0));

    if (new_obj) {
        auto* data = reinterpret_cast<RPySimpleTensorValuedStream*>(new_obj.
            ptr());
        construct_inplace(&data->p_data, std::move(ptr));
    }

    return new_obj;
}


extern "C" {
static PyObject* stvs_new(PyTypeObject* subtype,
                          PyObject* args,
                          PyObject* kwargs)
{
    static const char* kwlist[] = {
            "increment_stream", "initial_value", "domain", nullptr
    };

    python::RPyStream* incr_stream;
    PyObject* initial_value;
    PyObject* domain;

    if (PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O!O!O!",
                                    const_cast<char**>(kwlist),
                                    &python::RPyStream_Type,
                                    &incr_stream,
                                    py::type::of<FreeTensor>().ptr(),
                                    &initial_value,
                                    py::type::of<intervals::Interval>().ptr(),
                                    &domain
    )) { return nullptr; }

    auto new_obj = py::reinterpret_steal<py::object>(subtype->tp_alloc(subtype, 0));
    if (!new_obj) { return nullptr; }

    auto* self = reinterpret_cast<RPySimpleTensorValuedStream*>(new_obj.ptr());

    auto success = python::with_caught_exceptions([&]() {
        // Make sure the shared pointer is properly initialised
        construct_inplace(&self->p_data, make_simple_tensor_valued_stream(
                              incr_stream->m_data.impl(),
                              py::cast<const FreeTensor&>(initial_value),
                              py::cast<const intervals::Interval&>(domain)
                          ));
    });

    RPY_DBG_ASSERT(self->p_data || !success);

    return new_obj.release().ptr();
}

static void stvs_finalize(PyObject* self)
{
    auto& data = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    if (data) { data.reset(); }
    Py_TYPE(self)->tp_free(self);
}


static PyObject* stvs_query(PyObject* self, PyObject* py_domain)
{
    py::object result;
    try {
        const auto& domain = py::cast<const intervals::Interval&>(py_domain);
        const auto& vs = reinterpret_cast<const RPySimpleTensorValuedStream*>(
            self)->p_data;

        result = python::TensorValuedStream_FromPtr(vs->query(domain));
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }

    return result.release().ptr();
}


static PyObject* stvs_initial_value(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->initial_value());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}


static PyObject* stvs_terminal_value(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->terminal_value());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_increment_stream(PyObject* self)
{
    py::object result;

    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(Stream(stream->increment_stream())));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_domain(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->domain());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_repr(PyObject* self) { return PyObject_Repr(self); }

static PyObject* stvs_signature(PyObject* self,
                                PyObject* args,
                                PyObject* kwargs)
{
    python::SigArgs sig_args;

    const auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->
            p_data;
    if (parse_sig_args(args, kwargs, &stream->metadata(), &sig_args)
        < 0) { return nullptr; }

    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        if (!sig_args.interval) { sig_args.interval = stream->domain(); }
        if (!sig_args.resolution) {
            sig_args.resolution = stream->metadata().default_resolution;
        }
        if (!sig_args.ctx) { sig_args.ctx = stream->metadata().default_context; }

        algebra::FreeTensor sig;

        {
            py::gil_scoped_release gil;
            sig = stream->signature(*sig_args.interval,
                                    *sig_args.resolution,
                                    *sig_args.ctx);
        }

        result = py::cast(std::move(sig));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_log_signature(PyObject* self,
                                    PyObject* args,
                                    PyObject* kwargs)
{
    python::SigArgs sig_args;

    const auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->
            p_data;
    if (parse_sig_args(args, kwargs, &stream->metadata(), &sig_args)
        < 0) { return nullptr; }

    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        if (!sig_args.interval) { sig_args.interval = stream->domain(); }
        if (!sig_args.resolution) {
            sig_args.resolution = stream->metadata().default_resolution;
        }
        if (!sig_args.ctx) { sig_args.ctx = stream->metadata().default_context; }

        algebra::Lie logsig;
        {
            py::gil_scoped_release gil;
            logsig = stream->log_signature(*sig_args.interval,
                                           *sig_args.resolution,
                                           *sig_args.ctx);
        }

        result = py::cast(std::move(logsig));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}


}


// Define the method table for RPySimpleTensorValuedStream
static PyMethodDef stvs_methods[] = {
        {"query", reinterpret_cast<PyCFunction>(stvs_query), METH_O,
         "Query the stream with a given domain"},
        {"initial_value", reinterpret_cast<PyCFunction>(stvs_initial_value),
         METH_NOARGS, "Get the initial value of the stream"},
        {"terminal_value", reinterpret_cast<PyCFunction>(stvs_terminal_value),
         METH_NOARGS, "Get the terminal value of the stream"},
        {"increment_stream",
         reinterpret_cast<PyCFunction>(stvs_increment_stream), METH_NOARGS,
         "Get the increment stream"},
        {"domain", reinterpret_cast<PyCFunction>(stvs_domain), METH_NOARGS,
         "Get the domain of the stream"},
        {"signature", reinterpret_cast<PyCFunction>(stvs_signature),
         METH_VARARGS | METH_KEYWORDS, "Get the signature of the stream"},
        {"log_signature", reinterpret_cast<PyCFunction>(stvs_log_signature),
         METH_VARARGS | METH_KEYWORDS, "Get the log signature of the stream"},
        {nullptr, nullptr, 0, nullptr}// Sentinel
};

// Define the PyTypeObject for RPySimpleTensorValuedStream
PyTypeObject python::TensorValuedStream_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)
        "roughpy.TensorValuedStream",// tp_name
        sizeof(RPySimpleTensorValuedStream),// tp_basicsize
        0,// tp_itemsize
        0,// tp_dealloc
        0,// tp_vectorcall_offset
        0,// tp_getattr
        0,// tp_setattr
        0,// tp_as_async
        stvs_repr,// tp_repr
        0,// tp_as_number
        0,// tp_as_sequence
        0,// tp_as_mapping
        0,// tp_hash
        0,// tp_call
        0,// tp_str
        0,// tp_getattro
        0,// tp_setattro
        0,// tp_as_buffer
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_FINALIZE,
        // tp_flags
        "A Simple Tensor Valued stream",// tp_doc
        0,// tp_traverse
        0,// tp_clear
        0,// tp_richcompare
        0,// tp_weaklistoffset
        0,// tp_iter
        0,// tp_iternext
        stvs_methods,// tp_methods
        0,// tp_members
        0,// tp_getset
        0,// tp_base
        0,// tp_dict
        0,// tp_descr_get
        0,// tp_descr_set
        0,// tp_dictoffset
        0,// tp_init
        0,// tp_alloc
        stvs_new,// tp_new
        0,// tp_free
        0,// tp_is_gc
        0,// tp_bases
        0,// tp_mro
        0,// tp_cache
        0,// tp_subclasses
        0,// tp_weaklist
        0,// tp_del
        0,// tp_version_tag
        stvs_finalize// tp_finalize
};

void python::init_tensor_valued_stream(py::module_& m)
{
    if (PyType_Ready(&TensorValuedStream_Type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("TensorValuedStream",
                 reinterpret_cast<PyObject*>(&
                     TensorValuedStream_Type));

}