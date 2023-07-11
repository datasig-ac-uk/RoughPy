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

#include "context.h"

#include <pybind11/stl.h>

#include "lie_key_iterator.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "tensor_key_iterator.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;
using rpy::python::ctx_cast;
using rpy::python::RPyContext;

extern "C" {

static const char* lie_size_DOC = R"rpydoc()rpydoc";
static PyObject* RPyContext_lie_size(PyObject* self, PyObject* degree)
{
    auto deg = static_cast<deg_t>(PyLong_AsLong(degree));
    return PyLong_FromSize_t(ctx_cast(self)->lie_size(deg));
}
static const char* tensor_size_DOC = R"rpydoc()rpydoc";
static PyObject* RPyContext_tensor_size(PyObject* self, PyObject* degree)
{
    auto deg = static_cast<deg_t>(PyLong_AsLong(degree));
    return PyLong_FromSize_t(ctx_cast(self)->tensor_size(deg));
}
static const char* cbh_DOC = R"rpydoc()rpydoc";
static PyObject*
RPyContext_cbh(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[] = {"lies", "vec_type", nullptr};
    const auto& ctx = ctx_cast(self);

    PyObject* py_lies = nullptr;
    PyObject* vtype = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "O|O!", const_cast<char**>(kwords), &py_lies,
                py::type::of<VectorType>().ptr(), &vtype
        )) {
        return nullptr;
    }

    const auto num_lies = PySequence_Size(py_lies);
    std::vector<Lie> lies;
    lies.reserve(num_lies);

    for (Py_ssize_t i = 0; i < num_lies; ++i) {
        py::handle py_lie(PySequence_ITEM(py_lies, i));
        lies.push_back(py_lie.cast<Lie&>().borrow_mut());
    }

    if (lies.empty()) {
        if (vtype == nullptr) {
            return python::cast_to_object(ctx->zero_lie(VectorType::Sparse));
        }
        return python::cast_to_object(
                ctx->zero_lie(py::handle(vtype).cast<VectorType>())
        );
    }

    VectorType vtp = VectorType::Sparse;
    if (vtype != nullptr) { vtp = py::handle(vtype).cast<VectorType>(); }

    return python::cast_to_object(ctx->cbh(lies, vtp));
}
static const char* compute_signature_DOC = R"rpydoc()rpydoc";
static PyObject*
RPyContext_compute_signature(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[] = {"data", "vtype", nullptr};
    const auto& ctx = ctx_cast(self);

    py::handle py_data;
    py::handle py_vtype;

    if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "O|O!", const_cast<char**>(kwords), py_data.ptr(),
                py::type::of<VectorType>().ptr(), py_vtype.ptr()
        )) {
        return nullptr;
    }

    python::PyToBufferOptions options;
    options.allow_scalar = false;
    options.max_nested = 2;

    scalars::KeyScalarArray buffer;

    try {
        buffer = python::py_to_buffer(py_data, options);
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }

    SignatureData request;
    request.vector_type = VectorType::Sparse;
    if (py_vtype) { request.vector_type = py_vtype.cast<VectorType>(); }

    if (buffer.size() == 0) {
        auto result = ctx->zero_free_tensor(request.vector_type);
    }

    const auto* ctype = ctx->ctype();
    request.data_stream.set_ctype(ctype);
    const auto itemsize = ctype->itemsize();

    const auto* p_buffer = static_cast<const char*>(buffer.cptr());
    if (!buffer.has_keys()) {
        if (!py_vtype) { request.vector_type = VectorType::Dense; }

        if (options.shape.empty() || options.shape.size() > 2) {
            PyErr_SetString(PyExc_ValueError, "invalid shape");
            return nullptr;
        }

        dimn_t width;
        dimn_t n_increments;
        if (options.shape.size() == 1) {
            width = options.shape[0];
            n_increments = 1;
        } else {
            width = options.shape[1];
            n_increments = options.shape[0];
        }

        request.data_stream.set_elts_per_row(width);
        request.data_stream.reserve_size(n_increments);
        for (dimn_t i = 0; i < n_increments; ++i) {
            request.data_stream.push_back(
                    {ctype, p_buffer + i * width * itemsize}
            );
        }
    } else {
        request.data_stream.reserve_size(options.shape.size());
        request.key_stream.reserve(options.shape.size());
        const key_type* p_keys = buffer.keys();

        auto n_increments = options.shape.size();
        for (dimn_t i = 0; i < n_increments; ++i) {
            request.data_stream.push_back({
                    {ctype, p_buffer},
                    static_cast<dimn_t>(options.shape[i])
            });
            request.key_stream.push_back(p_keys);
            p_buffer += options.shape[i] * itemsize;
            p_keys += options.shape[i];
        }
    }

    return python::cast_to_object(ctx->signature(request));
}
static const char* to_logsignature_DOC = R"rpydoc()rpydoc";
static PyObject* RPyContext_to_logsignature(PyObject* self, PyObject* arg)
{

    py::handle py_sig(arg);
    if (!py::isinstance<algebra::FreeTensor>(py_sig)) {
        PyErr_SetString(PyExc_TypeError, "expected a FreeTensor object");
        return nullptr;
    }

    const auto& ctx = ctx_cast(self);

    const auto& sig = py_sig.cast<const FreeTensor&>();

    return python::cast_to_object(ctx->tensor_to_lie(sig.log()));
}
static const char* lie_to_tensor_DOC = R"rpydoc()rpydoc";
static PyObject* RPyContext_lie_to_tensor(PyObject* self, PyObject* arg)
{
    py::handle py_lie(arg);
    if (!py::isinstance<algebra::Lie>(py_lie)) {
        PyErr_SetString(PyExc_TypeError, "expected a Lie object");
        return nullptr;
    }

    const auto& ctx = ctx_cast(self);
    return python::cast_to_object(ctx->lie_to_tensor(py_lie.cast<const Lie&>())
    );
}
static const char* tensor_to_lie_DOC = R"rpydoc()rpydoc";
static PyObject* RPyContext_tensor_to_lie(PyObject* self, PyObject* arg)
{
    py::handle py_ft(arg);
    if (!py::isinstance<algebra::FreeTensor>(py_ft)) {
        PyErr_SetString(PyExc_TypeError, "expected a FreeTensor object");
        return nullptr;
    }

    const auto& ctx = ctx_cast(self);
    return python::cast_to_object(
            ctx->tensor_to_lie(py_ft.cast<const FreeTensor&>())
    );
}

static PyObject* RPyContext_enter(PyObject* self) { return self; }

static PyObject* RPyContext_exit(PyObject* self, PyObject* RPY_UNUSED_VAR)
{
    Py_RETURN_NONE;
}

static const char* zero_lie_DOC
        = R"rpydoc(Get a new Lie with value zero)rpydoc";
static PyObject*
RPyContext_zero_lie(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char* kwords[] = {"vtype", nullptr};
    PyTypeObject* vtype_type
            = reinterpret_cast<PyTypeObject*>(py::type::of<VectorType>().ptr());

    PyObject* py_vtype = nullptr;
    if (PyArg_ParseTupleAndKeywords(
                args, kwargs, "|O!", const_cast<char**>(kwords), vtype_type,
                &py_vtype
        )
        == 0) {
        return nullptr;
    }

    VectorType vtype = VectorType::Sparse;
    if (py_vtype != nullptr) { vtype = py::cast<VectorType>(py_vtype); }

    return python::cast_to_object(ctx_cast(self)->zero_lie(vtype));
}

#define ADD_METHOD(NAME, FLAGS)                                                \
    {                                                                          \
        #NAME, (PyCFunction) &RPyContext_##NAME, (FLAGS), NAME##_DOC           \
    }

static PyMethodDef RPyContext_members[] = {
        ADD_METHOD(lie_size, METH_O),
        ADD_METHOD(tensor_size, METH_O),
        ADD_METHOD(cbh, METH_VARARGS | METH_KEYWORDS),
        ADD_METHOD(compute_signature, METH_VARARGS | METH_KEYWORDS),
        ADD_METHOD(to_logsignature, METH_O),
        ADD_METHOD(lie_to_tensor, METH_O),
        ADD_METHOD(tensor_to_lie, METH_O),
        ADD_METHOD(zero_lie, METH_VARARGS | METH_KEYWORDS),
        {"__enter__", (PyCFunction) &RPyContext_enter,  METH_NOARGS, nullptr},
        { "__exit__",  (PyCFunction) &RPyContext_exit, METH_VARARGS, nullptr},
        {    nullptr,                         nullptr,            0, nullptr}
};

#undef ADD_METHOD

static PyObject* RPyContext_width_getter(PyObject* self)
{
    return PyLong_FromLong(ctx_cast(self)->width());
}
static PyObject* RPyContext_depth_getter(PyObject* self)
{
    return PyLong_FromLong(ctx_cast(self)->depth());
}
static PyObject* RPyContext_ctype_getter(PyObject* self)
{
    return python::to_ctype_type(ctx_cast(self)->ctype()).release().ptr();
}
static PyObject* RPyContext_lie_basis_getter(PyObject* self)
{
    return python::cast_to_object(ctx_cast(self)->get_lie_basis());
}
static PyObject* RPyContext_tensor_basis_getter(PyObject* self)
{
    return python::cast_to_object(ctx_cast(self)->get_tensor_basis());
}

#define ADD_GETSET(NAME)                                                       \
    {                                                                          \
        #NAME, (getter) &RPyContext_##NAME##_getter, nullptr, nullptr, nullptr \
    }

static PyGetSetDef RPyContext_getset[] = {
        ADD_GETSET(width),
        ADD_GETSET(depth),
        ADD_GETSET(ctype),
        ADD_GETSET(lie_basis),
        ADD_GETSET(tensor_basis),
        {nullptr, nullptr, nullptr, nullptr, nullptr}
};

#undef ADD_GETSET

static PyObject* RPyContext_repr(PyObject* self)
{
    const auto& ctx = ctx_cast(self);
    std::stringstream ss;
    ss << "Context(width=" << ctx->width() << ", depth=" << ctx->depth()
       << ", ctype=" << ctx->ctype()->info().name << ')';
    return PyUnicode_FromString(ss.str().c_str());
}
static PyObject* RPyContext_str(PyObject* self)
{
    return RPyContext_repr(self);
}

RPY_UNUSED
static PyObject*
RPyContext_new(PyObject* self, PyObject* args, PyObject* kwargs)
{
    RPY_UNREACHABLE_RETURN(nullptr);
}

static const char* CONTEXT_DOC = R"rpydoc()rpydoc";
PyTypeObject rpy::python::RPyContext_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0) "_roughpy.Context", /* tp_name */
        sizeof(python::RPyContext),                           /* tp_basicsize */
        0,                                                    /* tp_itemsize */
        nullptr,                                              /* tp_dealloc */
        0,                                        /* tp_vectorcall_offset */
        (getattrfunc) nullptr,                    /* tp_getattr */
        (setattrfunc) nullptr,                    /* tp_setattr */
        nullptr,                                  /* tp_as_async */
        (reprfunc) RPyContext_repr,               /* tp_repr */
        nullptr,                                  /* tp_as_number */
        nullptr,                                  /* tp_as_sequence */
        nullptr,                                  /* tp_as_mapping */
        nullptr,                                  /* tp_hash */
        nullptr,                                  /* tp_call */
        (reprfunc) RPyContext_str,                /* tp_str */
        nullptr,                                  /* tp_getattro */
        (setattrofunc) nullptr,                   /* tp_setattro */
        (PyBufferProcs*) nullptr,                 /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
        CONTEXT_DOC,                              /* tp_doc */
        nullptr,                                  /* tp_traverse */
        nullptr,                                  /* tp_clear */
        nullptr,                                  /* tp_richcompare */
        0,                                        /* tp_weaklistoffset */
        (getiterfunc) nullptr,                    /* tp_iter */
        nullptr,                                  /* tp_iternext */
        RPyContext_members,                       /* tp_methods */
        nullptr,                                  /* tp_members */
        RPyContext_getset,                        /* tp_getset */
        nullptr,                                  /* tp_base */
        nullptr,                                  /* tp_dict */
        nullptr,                                  /* tp_descr_get */
        nullptr,                                  /* tp_descr_set */
        0,                                        /* tp_dictoffset */
        (initproc) nullptr,                       /* tp_init */
        nullptr,                                  /* tp_alloc */
        PyType_GenericNew,                        /* tp_new */
};
}

// #ifdef ROUGHPY_WITH_NUMPY
// static FreeTensor context_compute_signature_numpy_darray(const
// python::PyContext &ctx,
//                                                          const
//                                                          py::array_t<double,
//                                                          py::array::forcecast>
//                                                          &array) {
//     assert(array.ndim() == 2);
//     auto shape = array.shape();
//
//     const auto n_increments = shape[0];
//     const auto width = shape[1];
//
//     assert(width == ctx->width());
//
//     const auto *ctype = ctx->ctype();
//     SignatureData request;
//     request.data_stream.set_ctype(ctype);
//     request.data_stream.set_elts_per_row(width);
//     request.data_stream.reserve_size(n_increments);
//     for (dimn_t i = 0; i < n_increments; ++i) {
//         request.data_stream.push_back(scalars::ScalarPointer(ctype,
//         array.data(i, 0)));
//     }
//     request.vector_type = VectorType::Dense;
//
//     auto sig = ctx->signature(request);
//
//     return sig;
//     //    return ctx->signature(request);
// }
// #endif

PyObject* python::RPyContext_FromContext(algebra::context_pointer ctx)
{
    auto* new_ctx = reinterpret_cast<RPyContext*>(
            RPyContext_Type.tp_alloc(&RPyContext_Type, 0)
    );
    new_ctx->p_ctx = std::move(ctx);
    return reinterpret_cast<PyObject*>(new_ctx);
}

static py::handle py_get_context(
        deg_t width, deg_t depth, const py::object& ctype,
        const py::kwargs& kwargs
)
{
    // TODO: Make this accept extra arguments.
    return python::RPyContext_FromContext(
            get_context(width, depth, python::to_stype_ptr(ctype), {})
    );
}

void python::init_context(py::module_& m)
{

    if (PyType_Ready(&RPyContext_Type) < 0) { throw py::error_already_set(); }
    m.add_object("Context", reinterpret_cast<PyObject*>(&RPyContext_Type));

    m.def("get_context", py_get_context, "width"_a, "depth"_a,
          "coeffs"_a = py::none());
}
