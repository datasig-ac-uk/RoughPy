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

//
// Created by user on 06/07/23.
//

#include "r_py_monomial.h"

#include <roughpy/core/alloc.h>

using namespace rpy;
using namespace python;

extern "C" {

// New
static PyObject*
monomial_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);

// finalize
static void monomial_finalize(PyObject* self);

// dealloc
static void monomial_dealloc(PyObject* self);

// Conversion
static int monomial_bool(PyObject* self);

// Hash
static Py_hash_t monomial_hash(PyObject* self);

// Comparison
static PyObject*
monomial_rich_compare(PyObject* self, PyObject* other, int cmp);

// Mapping
static Py_ssize_t monomial_len(PyObject* self);
static Py_ssize_t monomial_subscript(PyObject* self, PyObject* index);
static int
monomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg);

// Numbers
static PyObject* monomial_mul(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_mul(PyObject* self, PyObject* other);
static PyObject* monomial_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* monomial_divmod(PyObject* self, PyObject* other);
static PyObject* monomial_rem(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_rem(PyObject* self, PyObject* other);
static PyObject*
monomial_inplace_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* monomial_inplace_divmod(PyObject* self, PyObject* other);

static PyObject* monomial_add(PyObject* self, PyObject* other);
static PyObject* monomial_sub(PyObject* self, PyObject* other);
static PyObject* monomial_div(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_add(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_sub(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_div(PyObject* self, PyObject* other);
static PyObject* monomial_radd(PyObject* self, PyObject* other);
static PyObject* monomial_rsub(PyObject* self, PyObject* other);

// Printing
static PyObject* monomial_str(PyObject* self);
static PyObject* monomial_repr(PyObject* self);

// Other methods
static PyObject* monomial_degree(PyObject* self);
}

PyObject* PyMonomial_FromIndeterminate(scalars::indeterminate_type indet);
PyObject* PyMonomial_FromMonomial(scalars::monomial arg);

scalars::monomial PyMonomial_AsMonomial(PyObject* py_monomial);

scalars::indeterminate_type indeterminate_from_string(PyObject* py_string);
PyObject* PyMonomial_FromPyString(PyObject* py_string);

struct RPyMonomial {
    PyObject_VAR_HEAD scalars::monomial m_data;
};

static PyMethodDef PyMonomial_methods[] = {
        {"degree", (PyCFunction) &monomial_degree, METH_O, nullptr},
        { nullptr,                        nullptr,      0, nullptr}
};

static PyNumberMethods PyMonomial_number{
        (binaryfunc) monomial_add,            /* nb_add */
        (binaryfunc) monomial_sub,            /* nb_subtract */
        (binaryfunc) monomial_mul,            /* nb_multiply */
        (binaryfunc) monomial_rem,            /* nb_remainder */
        (binaryfunc) monomial_divmod,         /* nb_divmod */
        (ternaryfunc) monomial_pow,           /* nb_power */
        nullptr,                              /* nb_negative */
        nullptr,                              /* nb_positive */
        nullptr,                              /* nb_absolute */
        (inquiry) monomial_bool,              /* nb_bool */
        nullptr,                              /* nb_invert */
        nullptr,                              /* nb_lshift */
        nullptr,                              /* nb_rshift */
        nullptr,                              /* nb_and */
        nullptr,                              /* nb_xor */
        nullptr,                              /* nb_or */
        nullptr,                              /* nb_int */
        nullptr,                              /* nb_reserved */
        nullptr,                              /* nb_float */
        (binaryfunc) monomial_inplace_add,    /* nb_inplace_add */
        (binaryfunc) monomial_inplace_sub,    /* nb_inplace_subtract */
        (binaryfunc) monomial_inplace_mul,    /* nb_inplace_multiply */
        (binaryfunc) monomial_inplace_rem,    /* nb_inplace_remainder */
        (ternaryfunc) monomial_inplace_pow,   /* nb_inplace_power */
        nullptr,                              /* nb_inplace_lshift */
        nullptr,                              /* nb_inplace_rshift */
        nullptr,                              /* nb_inplace_and */
        nullptr,                              /* nb_inplace_xor */
        nullptr,                              /* nb_inplace_or */
        (binaryfunc) monomial_divmod,         /* nb_floor_divide */
        (binaryfunc) monomial_div,            /* nb_true_divide */
        (binaryfunc) monomial_inplace_divmod, /* nb_inplace_floor_divide */
        (binaryfunc) monomial_inplace_div,    /* nb_inplace_true_div */
        nullptr,                              /* nb_index */
        nullptr,                              /* nb_matrix_multiply */
        nullptr                               /* nb_inplace_matrix_multiply */
};

static PyTypeObject PyMonomial_type = {
        PyVarObject_HEAD_INIT(nullptr, 0) "_roughpy.Monomial", /* tp_name */
        sizeof(RPyMonomial),                 /* tp_basicsize */
        0,                                   /* tp_itemsize */
        (destructor) monomial_dealloc,       /* tp_dealloc */
        0,                                   /* tp_vectorcall_offset */
        0,                                   /* tp_getattr */
        0,                                   /* tp_setattr */
        0,                                   /* tp_as_async */
        (reprfunc) monomial_repr,            /* tp_repr */
        &PyMonomial_number,                  /* tp_as_number */
        0,                                   /* tp_as_sequence */
        0,                                   /* tp_as_mapping */
        (hashfunc) monomial_hash,            /* tp_hash */
        0,                                   /* tp_call */
        (reprfunc) monomial_str,             /* tp_str */
        0,                                   /* tp_getattro */
        0,                                   /* tp_setattro */
        0,                                   /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,                  /* tp_flags */
        PyDoc_STR("Monomial basis element"), /* tp_doc */
        0,                                   /* tp_traverse */
        0,                                   /* tp_clear */
        (richcmpfunc) monomial_rich_compare, /* tp_richcompare */
        0,                                   /* tp_weaklistoffset */
        0,                                   /* tp_iter */
        0,                                   /* tp_iternext */
        PyMonomial_methods,                  /* tp_methods */
        0,                                   /* tp_members */
        0,                                   /* tp_getset */
        0,                                   /* tp_base */
        0,                                   /* tp_dict */
        0,                                   /* tp_descr_get */
        0,                                   /* tp_descr_set */
        0,                                   /* tp_dictoffset */
        0,                                   /* tp_init */
        0,                                   /* tp_alloc */
        (newfunc) monomial_new,              /* tp_new */
        nullptr,                             /* tp_free */
        nullptr,                             /* tp_is_gc */
        nullptr,                             /* tp_bases */
        nullptr,                             /* tp_mro */
        nullptr,                             /* tp_cache */
        nullptr,                             /* tp_subclasses */
        nullptr,                             /* tp_weaklist */
        nullptr,                             /* tp_del */
        0,                                   /* tp_version_tag */
        monomial_finalize,                   /* tp_finalize */
        nullptr                              /* tp_vectorcall */
};

void python::init_monomial(py::module_& m)
{

    if (PyType_Ready(&PyMonomial_type) < 0) { throw py::error_already_set(); }

    m.add_object("Monomial", (PyObject*) &PyMonomial_type);
}


static bool insert_from_pair(scalars::monomial& monomial, PyObject* item) {
    auto* first = PyTuple_GetItem(item, 0);
    auto* second = PyTuple_GetItem(item, 1);

    if (!PyUnicode_Check(first) || !PyLong_Check(second)) {
        PyErr_SetString(
                PyExc_TypeError,
                "expected either a tuple (str, int)"
        );
        return false;
    }

    auto indet = indeterminate_from_string(first);
    auto pow = PyLong_AsLong(second);

    monomial *= scalars::monomial(indet, static_cast<deg_t>(pow));
}

static bool construct_new_monomial(RPyMonomial* p_obj, PyObject* args,
                                   PyObject* kwargs)
{
    PyObject* dict_def = nullptr;
    auto n_args = PyTuple_Size(args);
    if (n_args == 0) {
        if (kwargs == nullptr) {
            return true;
        }
        dict_def = kwargs;
    } else if (n_args == 1) {
        /*
         * If the number of arguments is 1 then the options are as follows:
         *  - a single string indeterminate;
         *  - a list of string indeterminates;
         *  - a list of (indeterminate, power) pairs
         *  - a dictionary of (indeterminate, power) pairs.
         */
        auto* arg = PyTuple_GetItem(args, 0);
        if (PyUnicode_Check(arg)) {
            auto indet = indeterminate_from_string(arg);
            p_obj->m_data *= scalars::monomial(indet);
            return true;
        } else if (PyDict_Check(arg)) {
            // Dict type handled below
            dict_def = arg;
        } else if (PySequence_Check(arg)) {
            auto size = PySequence_Size(arg);
            PyObject* item;

            for (Py_ssize_t i=0; i<size; ++i) {
                item = PySequence_GetItem(arg, i);

                if (PyUnicode_Check(item)) {
                    auto indet = indeterminate_from_string(item);
                    p_obj->m_data *= scalars::monomial(indet);
                } else if (PyTuple_Check(item) && PyTuple_Size(item) == 2) {
                    if (!insert_from_pair(p_obj->m_data, item)) {
                        return false;
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "expected either a str "
                                                     "or tuple (str, int)");
                    return false;
                }
            }

            return true;
        } else {
            PyErr_SetString(PyExc_TypeError, "expected either a str, dict, or "
                                             "sequence");
            return false;
        }
    } else if (n_args == 2) {
        return insert_from_pair(p_obj->m_data, args);
    } else {
        PyErr_SetString(PyExc_ValueError, "expected 0, 1, or 2 arguments");
        return false;
    }

    // The keyword arguments should define the monomial.
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(dict_def, &pos, &key, &value)) {
        auto letter = indeterminate_from_string(key);
        if (!PyLong_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "expected powers to be of int "
                                             "type");
            return false;
        }
        auto pow = PyLong_AsLong(value);
        p_obj->m_data *= scalars::monomial(letter, static_cast<deg_t>(pow));
    }
    return true;
}

PyObject* monomial_new(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    auto* p_obj = reinterpret_cast<RPyMonomial*>(type->tp_alloc(type, 0));
    if (!p_obj) {
        return nullptr;
    }

    // Python allocators don't initialise c++ objects, so use construct inplace
    // to build a new monomial object in the correct region of memory.
    construct_inplace(&p_obj->m_data);

    // Now parse the arguments and fill in the construction
    if (!construct_new_monomial(p_obj, args, kwargs)) {
        // Construction failed, make sure we deconstruct m_data
        p_obj->m_data.~monomial();
        Py_DECREF(p_obj);
        p_obj = nullptr;
    }

    return reinterpret_cast<PyObject*>(p_obj);
}
void monomial_finalize(PyObject* self) {
    // Make sure the destructor is run
    reinterpret_cast<RPyMonomial*>(self)->m_data.~monomial();
}
void monomial_dealloc(PyObject* self) {}
int monomial_bool(PyObject* self) { return 0; }
Py_hash_t monomial_hash(PyObject* self) { return 0; }
PyObject* monomial_rich_compare(PyObject* self, PyObject* other, int cmp)
{

    switch (cmp) {
        case Py_EQ: break;
        case Py_NE: break;
        case Py_LT: break;
        case Py_LE: break;
        case Py_GT: break;
        case Py_GE: break;
    }

    RPY_UNREACHABLE_RETURN(nullptr);
}
Py_ssize_t monomial_len(PyObject* self) { return 0; }
Py_ssize_t monomial_subscript(PyObject* self, PyObject* index) { return 0; }
int monomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg)
{
    return 0;
}
PyObject* monomial_mul(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_inplace_mul(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_pow(PyObject* self, PyObject* other, PyObject*)
{
    return nullptr;
}
PyObject* monomial_divmod(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_rem(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_inplace_rem(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_inplace_pow(PyObject* self, PyObject* other, PyObject*)
{
    return nullptr;
}
PyObject* monomial_inplace_divmod(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_add(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_sub(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_div(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_inplace_add(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_inplace_sub(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_inplace_div(PyObject* self, PyObject* other)
{
    return nullptr;
}
PyObject* monomial_radd(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_rsub(PyObject* self, PyObject* other) { return nullptr; }
PyObject* monomial_str(PyObject* self) { return nullptr; }
PyObject* monomial_repr(PyObject* self) { return nullptr; }
PyObject* monomial_degree(PyObject* self) { return nullptr; }
PyObject* PyMonomial_FromIndeterminate(scalars::indeterminate_type indet)
{
    return nullptr;
}
PyObject* PyMonomial_FromMonomial(scalars::monomial arg) { return nullptr; }
scalars::monomial PyMonomial_AsMonomial(PyObject* py_monomial)
{
    return rpy::scalars::monomial();
}
scalars::indeterminate_type indeterminate_from_string(PyObject* py_string)
{
    return rpy::scalars::indeterminate_type(0);
}
PyObject* PyMonomial_FromPyString(PyObject* py_string) { return nullptr; }
