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

#include "r_py_polynomial.h"

#include <sstream>

#include <roughpy/core/alloc.h>
#include <roughpy/scalars/scalar.h>

#include "scalars.h"

using namespace rpy;
using namespace python;

namespace {

using poly_t = scalars::rational_poly_scalar;
using rat_t = scalars::rational_scalar_type;

}// namespace

extern "C" {

// New
static PyObject*
monomial_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);

// finalize
static void monomial_finalize(PyObject* self);

// Conversion
static int monomial_bool(PyObject* self);

// Hash
static Py_hash_t monomial_hash(PyObject* self);

// Comparison
static PyObject*
monomial_rich_compare(PyObject* self, PyObject* other, int cmp);

// Mapping
static Py_ssize_t monomial_len(PyObject* self);
static PyObject* monomial_subscript(PyObject* self, PyObject* index);
static int
monomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg);

// Numbers
static PyObject* monomial_mul(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_mul(PyObject* self, PyObject* other);
static PyObject* monomial_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* monomial_floordiv(PyObject* self, PyObject* other);
static PyObject* monomial_rem(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_rem(PyObject* self, PyObject* other);
static PyObject*
monomial_inplace_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* monomial_inplace_floordiv(PyObject* self, PyObject* other);

static PyObject* monomial_neg(PyObject* self);
static PyObject* monomial_pos(PyObject* self);

static PyObject* monomial_add(PyObject* self, PyObject* other);
static PyObject* monomial_sub(PyObject* self, PyObject* other);
static PyObject* monomial_div(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_add(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_sub(PyObject* self, PyObject* other);
static PyObject* monomial_inplace_div(PyObject* self, PyObject* other);

// Printing
static PyObject* monomial_str(PyObject* self);
static PyObject* monomial_repr(PyObject* self);

// Other methods
static PyObject* monomial_degree(PyObject* self);
}

bool indeterminate_from_string(
        PyObject* py_string, scalars::indeterminate_type& out
);

PyObject* PyMonomial_FromPyString(PyObject* py_string);

static PyMethodDef RPyMonomial_methods[] = {
        {"degree", (PyCFunction) &monomial_degree, METH_O, nullptr},
        { nullptr,                        nullptr,      0, nullptr}
};

static PyMappingMethods RPyMonomial_mapping{
        monomial_len, monomial_subscript, monomial_ass_subscript};

static PyNumberMethods RPyMonomial_number{
        (binaryfunc) monomial_add,              /* nb_add */
        (binaryfunc) monomial_sub,              /* nb_subtract */
        (binaryfunc) monomial_mul,              /* nb_multiply */
        (binaryfunc) monomial_rem,              /* nb_remainder */
        nullptr,                                /* nb_divmod */
        (ternaryfunc) monomial_pow,             /* nb_power */
        monomial_neg,                           /* nb_negative */
        monomial_pos,                           /* nb_positive */
        nullptr,                                /* nb_absolute */
        (inquiry) monomial_bool,                /* nb_bool */
        nullptr,                                /* nb_invert */
        nullptr,                                /* nb_lshift */
        nullptr,                                /* nb_rshift */
        nullptr,                                /* nb_and */
        nullptr,                                /* nb_xor */
        nullptr,                                /* nb_or */
        nullptr,                                /* nb_int */
        nullptr,                                /* nb_reserved */
        nullptr,                                /* nb_float */
        (binaryfunc) monomial_inplace_add,      /* nb_inplace_add */
        (binaryfunc) monomial_inplace_sub,      /* nb_inplace_subtract */
        (binaryfunc) monomial_inplace_mul,      /* nb_inplace_multiply */
        (binaryfunc) monomial_inplace_rem,      /* nb_inplace_remainder */
        (ternaryfunc) monomial_inplace_pow,     /* nb_inplace_power */
        nullptr,                                /* nb_inplace_lshift */
        nullptr,                                /* nb_inplace_rshift */
        nullptr,                                /* nb_inplace_and */
        nullptr,                                /* nb_inplace_xor */
        nullptr,                                /* nb_inplace_or */
        (binaryfunc) monomial_floordiv,         /* nb_floor_divide */
        (binaryfunc) monomial_div,              /* nb_true_divide */
        (binaryfunc) monomial_inplace_floordiv, /* nb_inplace_floor_divide */
        (binaryfunc) monomial_inplace_div,      /* nb_inplace_true_div */
        nullptr,                                /* nb_index */
        nullptr,                                /* nb_matrix_multiply */
        nullptr                                 /* nb_inplace_matrix_multiply */
};

PyTypeObject RPyMonomial_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)    //
        "_roughpy.Monomial",                 /* tp_name */
        sizeof(RPyMonomial),                 /* tp_basicsize */
        0,                                   /* tp_itemsize */
        nullptr,                             /* tp_dealloc */
        0,                                   /* tp_vectorcall_offset */
        0,                                   /* tp_getattr */
        0,                                   /* tp_setattr */
        0,                                   /* tp_as_async */
        (reprfunc) monomial_repr,            /* tp_repr */
        &RPyMonomial_number,                 /* tp_as_number */
        0,                                   /* tp_as_sequence */
        &RPyMonomial_mapping,                /* tp_as_mapping */
        (hashfunc) monomial_hash,            /* tp_hash */
        0,                                   /* tp_call */
        (reprfunc) monomial_str,             /* tp_str */
        0,                                   /* tp_getattro */
        0,                                   /* tp_setattro */
        0,                                   /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,                  /* tp_flags */
        PyDoc_STR("Polynomial scalar type"), /* tp_doc */
        0,                                   /* tp_traverse */
        0,                                   /* tp_clear */
        (richcmpfunc) monomial_rich_compare, /* tp_richcompare */
        0,                                   /* tp_weaklistoffset */
        0,                                   /* tp_iter */
        0,                                   /* tp_iternext */
        RPyMonomial_methods,                 /* tp_methods */
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

extern "C" {

// New
static PyObject*
polynomial_new(PyTypeObject* type, PyObject* args, PyObject* kwargs);

// finalize
static void polynomial_finalize(PyObject* self);

// Conversion
static int polynomial_bool(PyObject* self);

// Comparison
static PyObject*
polynomial_rich_compare(PyObject* self, PyObject* other, int cmp);

// Mapping
static Py_ssize_t polynomial_len(PyObject* self);
static PyObject* polynomial_subscript(PyObject* self, PyObject* index);
static int
polynomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg);

// Numbers
static PyObject* polynomial_mul(PyObject* self, PyObject* other);
static PyObject* polynomial_inplace_mul(PyObject* self, PyObject* other);
static PyObject* polynomial_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* polynomial_floordiv(PyObject* self, PyObject* other);
static PyObject* polynomial_rem(PyObject* self, PyObject* other);
static PyObject* polynomial_inplace_rem(PyObject* self, PyObject* other);
static PyObject*
polynomial_inplace_pow(PyObject* self, PyObject* other, PyObject*);
static PyObject* polynomial_inplace_floordiv(PyObject* self, PyObject* other);

static PyObject* polynomial_neg(PyObject* self);
static PyObject* polynomial_pos(PyObject* self);

static PyObject* polynomial_add(PyObject* self, PyObject* other);
static PyObject* polynomial_sub(PyObject* self, PyObject* other);
static PyObject* polynomial_div(PyObject* self, PyObject* other);
static PyObject* polynomial_inplace_add(PyObject* self, PyObject* other);
static PyObject* polynomial_inplace_sub(PyObject* self, PyObject* other);
static PyObject* polynomial_inplace_div(PyObject* self, PyObject* other);

// Printing
static PyObject* polynomial_str(PyObject* self);
static PyObject* polynomial_repr(PyObject* self);

// Other methods
static PyObject* polynomial_degree(PyObject* self);
}

PyObject* PyPolynomial_FromPolynomial(scalars::rational_poly_scalar&& poly
) noexcept;

static PyMethodDef RPyPolynomial_methods[] = {
        {"degree", (PyCFunction) &polynomial_degree, METH_O, nullptr},
        { nullptr,                          nullptr,      0, nullptr}
};

static PyMappingMethods RPyPolynomial_mapping{
        polynomial_len, polynomial_subscript, polynomial_ass_subscript};

static PyNumberMethods RPyPolynomial_number{
        (binaryfunc) polynomial_add,              /* nb_add */
        (binaryfunc) polynomial_sub,              /* nb_subtract */
        (binaryfunc) polynomial_mul,              /* nb_multiply */
        (binaryfunc) polynomial_rem,              /* nb_remainder */
        nullptr,                                  /* nb_divmod */
        (ternaryfunc) polynomial_pow,             /* nb_power */
        polynomial_neg,                           /* nb_negative */
        polynomial_pos,                           /* nb_positive */
        nullptr,                                  /* nb_absolute */
        polynomial_bool,                          /* nb_bool */
        nullptr,                                  /* nb_invert */
        nullptr,                                  /* nb_lshift */
        nullptr,                                  /* nb_rshift */
        nullptr,                                  /* nb_and */
        nullptr,                                  /* nb_xor */
        nullptr,                                  /* nb_or */
        nullptr,                                  /* nb_int */
        nullptr,                                  /* nb_reserved */
        nullptr,                                  /* nb_float */
        (binaryfunc) polynomial_inplace_add,      /* nb_inplace_add */
        (binaryfunc) polynomial_inplace_sub,      /* nb_inplace_subtract */
        (binaryfunc) polynomial_inplace_mul,      /* nb_inplace_multiply */
        (binaryfunc) polynomial_inplace_rem,      /* nb_inplace_remainder */
        (ternaryfunc) polynomial_inplace_pow,     /* nb_inplace_power */
        nullptr,                                  /* nb_inplace_lshift */
        nullptr,                                  /* nb_inplace_rshift */
        nullptr,                                  /* nb_inplace_and */
        nullptr,                                  /* nb_inplace_xor */
        nullptr,                                  /* nb_inplace_or */
        (binaryfunc) polynomial_floordiv,         /* nb_floor_divide */
        (binaryfunc) polynomial_div,              /* nb_true_divide */
        (binaryfunc) polynomial_inplace_floordiv, /* nb_inplace_floor_divide */
        (binaryfunc) polynomial_inplace_div,      /* nb_inplace_true_div */
        nullptr,                                  /* nb_index */
        nullptr,                                  /* nb_matrix_multiply */
        nullptr /* nb_inplace_matrix_multiply */
};

PyTypeObject RPyPolynomial_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)      //
        "_roughpy.Polynomial",                 /* tp_name */
        sizeof(RPyPolynomial),                 /* tp_basicsize */
        0,                                     /* tp_itemsize */
        nullptr,                               /* tp_dealloc */
        0,                                     /* tp_vectorcall_offset */
        0,                                     /* tp_getattr */
        0,                                     /* tp_setattr */
        0,                                     /* tp_as_async */
        (reprfunc) polynomial_repr,            /* tp_repr */
        &RPyPolynomial_number,                 /* tp_as_number */
        0,                                     /* tp_as_sequence */
        &RPyPolynomial_mapping,                /* tp_as_mapping */
        nullptr,                               /* tp_hash */
        0,                                     /* tp_call */
        (reprfunc) polynomial_str,             /* tp_str */
        0,                                     /* tp_getattro */
        0,                                     /* tp_setattro */
        0,                                     /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,                    /* tp_flags */
        PyDoc_STR("Monomial basis element"),   /* tp_doc */
        0,                                     /* tp_traverse */
        0,                                     /* tp_clear */
        (richcmpfunc) polynomial_rich_compare, /* tp_richcompare */
        0,                                     /* tp_weaklistoffset */
        0,                                     /* tp_iter */
        0,                                     /* tp_iternext */
        RPyPolynomial_methods,                 /* tp_methods */
        0,                                     /* tp_members */
        0,                                     /* tp_getset */
        0,                                     /* tp_base */
        0,                                     /* tp_dict */
        0,                                     /* tp_descr_get */
        0,                                     /* tp_descr_set */
        0,                                     /* tp_dictoffset */
        0,                                     /* tp_init */
        0,                                     /* tp_alloc */
        (newfunc) polynomial_new,              /* tp_new */
        nullptr,                               /* tp_free */
        nullptr,                               /* tp_is_gc */
        nullptr,                               /* tp_bases */
        nullptr,                               /* tp_mro */
        nullptr,                               /* tp_cache */
        nullptr,                               /* tp_subclasses */
        nullptr,                               /* tp_weaklist */
        nullptr,                               /* tp_del */
        0,                                     /* tp_version_tag */
        polynomial_finalize,                   /* tp_finalize */
        nullptr                                /* tp_vectorcall */
};

void python::init_monomial(py::module_& m)
{

    if (PyType_Ready(&RPyMonomial_Type) < 0) { throw py::error_already_set(); }
    if (PyType_Ready(&RPyPolynomial_Type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("Monomial", (PyObject*) &RPyMonomial_Type);
    m.add_object("PolynomialScalar", (PyObject*) &RPyPolynomial_Type);
}

static inline bool is_monomial(PyObject* obj) noexcept
{
    return Py_TYPE(obj) == &RPyMonomial_Type;
}
static inline bool is_polynomial(PyObject* obj) noexcept
{
    return Py_TYPE(obj) == &RPyPolynomial_Type;
}
static inline scalars::monomial& cast_mon(PyObject* obj) noexcept
{
    return reinterpret_cast<RPyMonomial*>(obj)->m_data;
}
static inline poly_t& cast_poly(PyObject* obj) noexcept
{
    return reinterpret_cast<RPyPolynomial*>(obj)->m_data;
}

static bool insert_from_pair(scalars::monomial& monomial, PyObject* item)
{
    auto* first = PyTuple_GetItem(item, 0);
    auto* second = PyTuple_GetItem(item, 1);

    if (!PyUnicode_Check(first) || !PyLong_Check(second)) {
        PyErr_SetString(PyExc_TypeError, "expected either a tuple (str, int)");
        return false;
    }

    scalars::indeterminate_type indet(0, 0);

    if (!indeterminate_from_string(first, indet)) { return false; }
    auto pow = PyLong_AsLong(second);

    monomial *= scalars::monomial(indet, static_cast<deg_t>(pow));
    return true;
}

static bool
construct_new_monomial(RPyMonomial* p_obj, PyObject* args, PyObject* kwargs)
{
    PyObject* dict_def = nullptr;
    auto n_args = PyTuple_Size(args);
    if (n_args == 0) {
        if (kwargs == nullptr) { return true; }
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
        scalars::indeterminate_type indet(0, 0);
        if (PyUnicode_Check(arg)) {
            if (!indeterminate_from_string(arg, indet)) { return false; }
            p_obj->m_data *= scalars::monomial(indet);
            return true;
        } else if (PyDict_Check(arg)) {
            // Dict type handled below
            dict_def = arg;
        } else if (PySequence_Check(arg)) {
            auto size = PySequence_Size(arg);
            PyObject* item;

            for (Py_ssize_t i = 0; i < size; ++i) {
                item = PySequence_GetItem(arg, i);

                if (PyUnicode_Check(item)) {
                    if (!indeterminate_from_string(item, indet)) {
                        return false;
                    }
                    p_obj->m_data *= scalars::monomial(indet);
                } else if (PyTuple_Check(item) && PyTuple_Size(item) == 2) {
                    if (!insert_from_pair(p_obj->m_data, item)) {
                        return false;
                    }
                } else {
                    PyErr_SetString(
                            PyExc_TypeError,
                            "expected either a str "
                            "or tuple (str, int)"
                    );
                    return false;
                }
            }

            return true;
        } else {
            PyErr_SetString(
                    PyExc_TypeError,
                    "expected either a str, dict, or "
                    "sequence"
            );
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
    scalars::indeterminate_type letter(0, 0);

    while (PyDict_Next(dict_def, &pos, &key, &value)) {
        if (!indeterminate_from_string(key, letter)) { return false; }
        if (!PyLong_Check(value)) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "expected powers to be of int "
                    "type"
            );
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
    if (p_obj == nullptr) { return nullptr; }

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
void monomial_finalize(PyObject* self)
{
    // Make sure the destructor is run
    reinterpret_cast<RPyMonomial*>(self)->m_data.~monomial();
}
int monomial_bool(PyObject* self)
{
    const auto& obj = reinterpret_cast<RPyMonomial*>(self)->m_data;
    return static_cast<int>(obj.type() > 0);
}
Py_hash_t monomial_hash(PyObject* self)
{
    hash<scalars::monomial> hasher;
    auto hash_val = static_cast<Py_hash_t>(hasher(cast_mon(self)));
    return hash_val;
}
PyObject* monomial_rich_compare(PyObject* self, PyObject* other, int cmp)
{
    if (is_monomial(self) && is_monomial(other)) {
        const auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);

        switch (cmp) {
            case Py_EQ: return PyBool_FromLong(static_cast<long>(lhs == rhs));
            case Py_NE:
                return PyBool_FromLong(static_cast<long>(!(lhs == rhs)));
            case Py_LT: return PyBool_FromLong(static_cast<long>(lhs < rhs));
            case Py_LE:
                return PyBool_FromLong(
                        static_cast<long>(lhs < rhs || lhs == rhs)
                );
            case Py_GT: return PyBool_FromLong(static_cast<long>(rhs < lhs));
            case Py_GE:
                return PyBool_FromLong(
                        static_cast<long>(rhs < lhs || lhs == rhs)
                );
            default: RPY_UNREACHABLE();
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
Py_ssize_t monomial_len(PyObject* self) { return cast_mon(self).type(); }
PyObject* monomial_subscript(PyObject* self, PyObject* index)
{
    const auto& mon = cast_mon(self);
    if (!PyUnicode_Check(index)) {
        PyErr_SetString(PyExc_TypeError, "index should be a string");
        return nullptr;
    }

    scalars::indeterminate_type letter(0, 0);
    if (!indeterminate_from_string(index, letter)) { return nullptr; }

    return PyLong_FromLong(mon[letter]);
}
int monomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg)
{
    auto& mon = cast_mon(self);
    if (!PyUnicode_Check(index)) {
        PyErr_SetString(PyExc_TypeError, "index should be a string");
        return -1;
    }
    if (!PyLong_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "expected an int");
        return -1;
    }

    scalars::indeterminate_type letter(0, 0);
    if (!indeterminate_from_string(index, letter)) { return -1; }

    try {
        mon[letter] = static_cast<deg_t>(PyLong_AsLong(arg));
    } catch (std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, "an unexpected error occurred");
        return -1;
    }

    return 0;
}

static inline bool py_scalar_to_rat(rat_t& result, PyObject* scalar) noexcept
{
    if (PyLong_Check(scalar)) {
        result = rat_t(PyLong_AsLongLong(scalar));
        return true;
    }
    if (PyFloat_Check(scalar)) {
        result = rat_t(PyFloat_AsDouble(scalar));
        return true;
    }

    try {
        auto fraction = get_py_rational();
        if (py::isinstance(scalar, fraction)) {
            auto numr = PyLong_AsLongLong(
                    PyObject_GetAttrString(scalar, "numerator")
            );
            auto denr = PyLong_AsLongLong(
                    PyObject_GetAttrString(scalar, "denominator")
            );

            result = rat_t(numr, denr);
            return true;
        }
    } catch (py::error_already_set&) {
        // If the error state is set, then clear it. This isn't really an
        // error.
        PyErr_Clear();
    }

    return false;
}

PyObject* monomial_mul(PyObject* self, PyObject* other)
{
    bool self_mon = is_monomial(self);
    if (self_mon && is_monomial(other)) {
        const auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);
        return PyMonomial_FromMonomial(lhs * rhs);
    }

    rat_t scalar;
    if (self_mon) {
        if (py_scalar_to_rat(scalar, other)) {
            return PyPolynomial_FromPolynomial(poly_t(cast_mon(self), scalar));
        }
    } else {
        if (py_scalar_to_rat(scalar, self)) {
            return PyPolynomial_FromPolynomial(poly_t(cast_mon(other), scalar));
        }
    }

    // We might want to support other types in the future.

    // If one of the operands is not a monomial, then the result will be a
    // polynomial.
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_mul(PyObject* self, PyObject* other)
{
    bool self_mon = is_monomial(self);

    if (self_mon && is_monomial(other)) {
        auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);
        lhs *= rhs;
        return self;
    }

    rat_t scalar;
    if (self_mon) {
        if (py_scalar_to_rat(scalar, other)) {
            return PyPolynomial_FromPolynomial(poly_t(cast_mon(self), scalar));
        }
    } else {
        if (py_scalar_to_rat(scalar, self)) {
            return PyPolynomial_FromPolynomial(poly_t(cast_mon(other), scalar));
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_pow(PyObject* self, PyObject* other, PyObject*)
{
    if (is_monomial(self) && PyLong_Check(other)) {
        const auto& lhs = cast_mon(self);
        auto power = static_cast<deg_t>(PyLong_AsLong(other));

        if (power < 0) {
            PyErr_SetString(PyExc_ValueError, "powers cannot be negative");
            return nullptr;
        }

        auto* result = PyMonomial_FromMonomial(lhs);
        auto& result_mon = cast_mon(result);
        for (auto& comp : result_mon) { comp.second += power; }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_floordiv(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_rem(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_rem(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_pow(PyObject* self, PyObject* other, PyObject*)
{
    if (is_monomial(self) && PyLong_Check(other)) {
        auto& lhs = cast_mon(self);
        auto power = static_cast<deg_t>(PyLong_AsLong(other));

        if (power < 0) {
            PyErr_SetString(PyExc_ValueError, "powers cannot be negative");
            return nullptr;
        }

        for (auto& comp : lhs) { comp.second += power; }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_floordiv(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject* monomial_neg(PyObject* self)
{
    return PyPolynomial_FromPolynomial(poly_t(cast_mon(self), rat_t(-1)));
}
PyObject* monomial_pos(PyObject* self)
{
    return PyPolynomial_FromPolynomial(poly_t(cast_mon(self), rat_t(1)));
}

PyObject* monomial_add(PyObject* self, PyObject* other)
{

    bool self_mon = is_monomial(self);
    if (self_mon && is_monomial(other)) {
        const auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);

        scalars::rational_poly_scalar result(
                lhs, scalars::rational_scalar_type(1)
        );
        result[rhs] += scalars::rational_scalar_type(1);
        return PyPolynomial_FromPolynomial(std::move(result));
    }

    rat_t scalar;
    if (self_mon) {
        if (py_scalar_to_rat(scalar, other)) {
            poly_t result(cast_mon(self), rat_t(1));
            result += poly_t(scalar);
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    } else {
        if (py_scalar_to_rat(scalar, self)) {
            poly_t result(scalar);
            result += poly_t(cast_mon(other), rat_t(1));
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_sub(PyObject* self, PyObject* other)
{
    bool self_mon = is_monomial(self);
    if (is_monomial(self) && is_monomial(other)) {
        const auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);

        scalars::rational_poly_scalar result(
                lhs, scalars::rational_scalar_type(1)
        );
        result[rhs] -= scalars::rational_scalar_type(1);
        return PyPolynomial_FromPolynomial(std::move(result));
    }

    rat_t scalar;
    if (self_mon) {
        if (py_scalar_to_rat(scalar, other)) {
            poly_t result(cast_mon(self), rat_t(1));
            result -= poly_t(scalar);
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    } else {
        if (py_scalar_to_rat(scalar, self)) {
            poly_t result(scalar);
            result -= poly_t(cast_mon(other), rat_t(1));
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_div(PyObject* self, PyObject* other)
{
    /*
     * Only the left hand argument can be a monomial, in which case it is a
     * monomial / scalar type operation.
     */
    if (!is_monomial(self) || is_monomial(other)) { Py_RETURN_NOTIMPLEMENTED; }

    rat_t scalar;
    if (py_scalar_to_rat(scalar, other)) {
        if (scalar == rat_t(0)) {
            PyErr_SetString(PyExc_ArithmeticError, "division by zero");
            return nullptr;
        }
        scalar = rat_t(1) / scalar;
        return PyPolynomial_FromPolynomial(poly_t(cast_mon(self), scalar));
    }

    // We might want to support other types in the future
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_add(PyObject* self, PyObject* other)
{
    bool self_mon = is_monomial(self);
    if (self_mon && is_monomial(other)) {
        auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);

        poly_t result(lhs, rat_t(1));
        result[rhs] += rat_t(1);
        return PyPolynomial_FromPolynomial(std::move(result));
    }

    rat_t scalar;
    if (self_mon) {
        const auto& mon = cast_mon(self);
        if (py_scalar_to_rat(scalar, other)) {
            poly_t result(mon, rat_t(1));
            result += poly_t(scalar);
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    } else {
        const auto& mon = cast_mon(other);
        if (py_scalar_to_rat(scalar, self)) {
            poly_t result(scalar);
            result += poly_t(mon, rat_t(1));
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    }

    // We might want to support other types in the future.

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_sub(PyObject* self, PyObject* other)
{
    bool self_mon = is_monomial(self);

    if (self_mon && is_monomial(other)) {
        const auto& lhs = cast_mon(self);
        const auto& rhs = cast_mon(other);

        poly_t result(lhs, scalars::rational_scalar_type(1));
        result[rhs] -= scalars::rational_scalar_type(1);
        return PyPolynomial_FromPolynomial(std::move(result));
    }

    if (self_mon) {
        const auto& mon = cast_mon(self);

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            poly_t result(mon, rat_t(1));
            result -= poly_t(scalar);
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    } else {
        const auto& mon = cast_mon(other);

        rat_t scalar;
        if (py_scalar_to_rat(scalar, self)) {
            poly_t result(scalar);
            result.add_scal_prod(mon, rat_t(1));
            return PyPolynomial_FromPolynomial(std::move(result));
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_inplace_div(PyObject* self, PyObject* other)
{
    /*
     * Only the left hand argument can be a monomial, in which case it is a
     * monomial / scalar type operation.
     */
    if (!is_monomial(self) || is_monomial(other)) { Py_RETURN_NOTIMPLEMENTED; }

    const auto& mon = cast_mon(self);

    rat_t scalar;
    if (py_scalar_to_rat(scalar, other)) {
        if (scalar == rat_t(0)) {
            PyErr_SetString(PyExc_ArithmeticError, "division by zero");
            return nullptr;
        }

        rat_t divisor = rat_t(1) / scalar;
        return PyPolynomial_FromPolynomial(poly_t(mon, scalar));
    }

    // We might want to support other types in the future
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* monomial_str(PyObject* self)
{
    const auto& obj = cast_mon(self);
    std::stringstream ss;
    ss << obj;
    return PyUnicode_FromString(ss.str().c_str());
}
PyObject* monomial_repr(PyObject* self)
{
    const auto& obj = cast_mon(self);
    std::stringstream ss;
    ss << obj;
    return PyUnicode_FromFormat("Monomial(%s)", ss.str().c_str());
}
PyObject* monomial_degree(PyObject* self)
{
    PyErr_SetString(PyExc_NotImplementedError, "not implemented");
    return nullptr;
}
PyObject* PyMonomial_FromIndeterminate(scalars::indeterminate_type indet)
{
    auto* obj = reinterpret_cast<RPyMonomial*>(
            RPyMonomial_Type.tp_alloc(&RPyMonomial_Type, 0)
    );

    if (obj == nullptr) { return nullptr; }

    construct_inplace(&obj->m_data, indet);
    return reinterpret_cast<PyObject*>(obj);
}
PyObject* PyMonomial_FromMonomial(scalars::monomial arg)
{
    auto* obj = reinterpret_cast<RPyMonomial*>(
            RPyMonomial_Type.tp_alloc(&RPyMonomial_Type, 0)
    );

    if (obj == nullptr) { return nullptr; }

    construct_inplace(&obj->m_data, std::move(arg));

    return reinterpret_cast<PyObject*>(obj);
}
scalars::monomial PyMonomial_AsMonomial(PyObject* py_monomial)
{
    if (Py_TYPE(py_monomial) != &RPyMonomial_Type) {
        RPY_THROW(std::invalid_argument, "");
    }

    return cast_mon(py_monomial);
}
bool indeterminate_from_string(
        PyObject* py_string, scalars::indeterminate_type& out
)
{
    const auto size = PyUnicode_GET_LENGTH(py_string);
    if (size == 0) {
        PyErr_SetString(
                PyExc_ValueError, "cannot parse indeterminate from empty string"
        );
        return false;
    }

    auto chr = PyUnicode_READ_CHAR(py_string, 0);
    if (!Py_UNICODE_ISALPHA(chr)) {
        PyErr_SetString(
                PyExc_ValueError,
                "expected first letter to be an latin character"
        );
        return false;
    }
    const char sym = static_cast<char>(chr);

    using int_type = typename scalars::indeterminate_type ::integral_type;

    int_type index = 0;
    for (Py_ssize_t i = 1; i < size; ++i) {
        chr = PyUnicode_READ_CHAR(py_string, i);
        if (!Py_UNICODE_ISDIGIT(chr)) {
            PyErr_SetString(PyExc_ValueError, "expected a digit");
            return false;
        }
        index *= 10;
        index += static_cast<int_type>(Py_UNICODE_TODIGIT(chr));
    }

    out = scalars::indeterminate_type(sym, index);
    return true;
}
PyObject* PyMonomial_FromPyString(PyObject* py_string)
{
    if (!PyUnicode_Check(py_string)) {
        PyErr_SetString(PyExc_TypeError, "expected a str");
        return nullptr;
    }

    scalars::indeterminate_type indet(0, 0);
    if (!indeterminate_from_string(py_string, indet)) { return nullptr; }

    auto* obj = reinterpret_cast<RPyMonomial*>(
            RPyMonomial_Type.tp_alloc(&RPyMonomial_Type, 0)
    );

    construct_inplace(&obj->m_data, indet);

    return reinterpret_cast<PyObject*>(obj);
}

static inline bool fill_poly_from_dict(poly_t& result, PyObject* dict_data)
{
    PyObject* key;
    PyObject* value;
    Py_ssize_t i = 0;

    rat_t scalar;
    bool parsed;
    while (PyDict_Next(dict_data, &i, &key, &value)) {
        parsed = py_scalar_to_rat(scalar, value);
        if (is_monomial(key) && parsed) {
            result += poly_t(cast_mon(key), scalar);
        } else {
            PyErr_SetString(
                    PyExc_TypeError, "expected monomial -> scalar mapping"
            );
            return false;
        }
    }

    return true;
}

PyObject* polynomial_new(PyTypeObject* type, PyObject* args, PyObject* kwargs)
{
    poly_t result;

    auto nargs = PyTuple_Size(args);
    rat_t scalar;
    if (nargs == 1) {
        /*
         * Single argument can be a scalar, a monomial, or a dict.
         */
        auto* arg = PyTuple_GetItem(args, 0);
        Py_INCREF(arg);
        if (PyDict_Check(arg)) {
            if (!fill_poly_from_dict(result, arg)) {
                Py_DECREF(arg);
                return nullptr;
            }
        } else if (is_monomial(arg)) {
            const auto& monomial = cast_mon(arg);
            try {
                result = poly_t(monomial, rat_t(1));
            } catch (std::exception& exc) {
                Py_DECREF(arg);
                PyErr_SetString(PyExc_RuntimeError, exc.what());
                return nullptr;
            }
        } else if (py_scalar_to_rat(scalar, arg)) {
            result = poly_t(std::move(scalar));
        } else {
            Py_DECREF(arg);
            PyErr_SetString(
                    PyExc_ValueError, "expected scalar, monomial, or dict"
            );
            return nullptr;
        }
        Py_DECREF(arg);
    } else if (nargs == 2) {
        /*
         * Two arguments means monomial/scalar pair.
         */
        auto* monomial = PyTuple_GetItem(args, 0);
        Py_INCREF(monomial);
        auto* scalar = PyTuple_GetItem(args, 1);
        Py_INCREF(scalar);

        if (!is_monomial(monomial)) {
            Py_DECREF(monomial);
            Py_DECREF(scalar);
            PyErr_SetString(
                    PyExc_TypeError, "expected a monomial as the first argument"
            );
            return nullptr;
        }

        const auto& mon = cast_mon(monomial);
        rat_t scal;
        if (!py_scalar_to_rat(scal, scalar)) {
            Py_DECREF(monomial);
            Py_DECREF(scalar);
            PyErr_SetString(
                    PyExc_TypeError, "expected a scalar as the second argument"
            );
            return nullptr;
        }
        Py_DECREF(monomial);
        Py_DECREF(scalar);

        if (scal != 0) { result = poly_t(mon, scal); }
    } else {
        PyErr_SetString(PyExc_ValueError, "execpted one or 2 arguments");
        return nullptr;
    }

    return PyPolynomial_FromPolynomial(std::move(result));
}
void polynomial_finalize(PyObject* self) { cast_poly(self).~polynomial(); }
PyObject* polynomial_rich_compare(PyObject* self, PyObject* other, int cmp)
{
    if (is_polynomial(self) && is_polynomial(other)) {
        const auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);

        switch (cmp) {
            case Py_EQ: return PyBool_FromLong(static_cast<long>(lhs == rhs));
            case Py_NE: return PyBool_FromLong(static_cast<long>(lhs != rhs));
            case Py_LT:
            case Py_LE:
            case Py_GE:
            case Py_GT: Py_RETURN_NOTIMPLEMENTED;
            default: RPY_UNREACHABLE();
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
Py_ssize_t polynomial_len(PyObject* self)
{
    const auto& obj = cast_poly(self);
    return static_cast<Py_ssize_t>(obj.size());
}
int polynomial_bool(PyObject* self)
{
    return static_cast<int>(!cast_poly(self).empty());
}
PyObject* polynomial_subscript(PyObject* self, PyObject* index)
{
    const auto& poly = cast_poly(self);
    if (is_monomial(index)) {
        return py::cast(scalars::Scalar(
                                scalars::ScalarType::of<rat_t>(),
                                poly[cast_mon(index)]
                        ))
                .release()
                .ptr();
    }

    PyErr_SetString(
            PyExc_TypeError,
            "polynomial index must be monomial or "
            "str"
    );
    return nullptr;
}
int polynomial_ass_subscript(PyObject* self, PyObject* index, PyObject* arg)
{
    auto& poly = cast_poly(self);
    scalars::monomial monomial;

    if (is_monomial(index)) {
        monomial = cast_mon(index);
    } else if (PyUnicode_Check(index)) {
        scalars::indeterminate_type indet(0, 0);
        if (!indeterminate_from_string(index, indet)) {
            PyErr_SetString(
                    PyExc_ValueError,
                    "could not parse string to "
                    "monomial"
            );
            return -1;
        }

        monomial = scalars::monomial(indet);
    }

    rat_t newarg;
    if (py_scalar_to_rat(newarg, arg)) {
        PyErr_SetString(PyExc_ValueError, "could not parse value");
        return -1;
    }

    if (newarg != rat_t(0)) {
        poly[monomial] = newarg;
        return 0;
    }

    PyErr_SetString(
            PyExc_TypeError, "polynomial index must be monomial or str"
    );
    return -1;
}
PyObject* polynomial_mul(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        const auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        return PyPolynomial_FromPolynomial(lhs * rhs);
    }

    if (lhs_poly) {
        const auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            return PyPolynomial_FromPolynomial(poly * poly_t(mon, rat_t(1)));
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            return PyPolynomial_FromPolynomial(poly * poly_t(scalar));
        }
    } else {
        const auto& poly = cast_poly(other);

        if (is_monomial(self)) {
            const auto& mon = cast_mon(self);
            return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) * poly);
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, self)) {
            return PyPolynomial_FromPolynomial(poly_t(scalar) * poly);
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_mul(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        lhs *= rhs;
        return self;
    }

    if (lhs_poly) {
        auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            poly *= poly_t(mon, rat_t(1));
            return self;
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            poly *= poly_t(scalar);
            return self;
        }

        Py_RETURN_NOTIMPLEMENTED;
    }

    const auto& poly = cast_poly(other);

    if (is_monomial(self)) {
        const auto& mon = cast_mon(self);
        return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) * poly);
    }

    rat_t scalar;
    if (py_scalar_to_rat(scalar, self)) {
        return PyPolynomial_FromPolynomial(poly_t(scalar) * poly);
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_pow(PyObject* self, PyObject* other, PyObject*)
{
    const auto& poly = cast_poly(self);
    if (!PyLong_Check(other)) { Py_RETURN_NOTIMPLEMENTED; }

    long power = PyLong_AsLong(other);
    if (power < 0) {
        PyErr_SetString(
                PyExc_ValueError, "cannot raise poly to negative power"
        );
        return nullptr;
    }

    if (power == 0) { return PyPolynomial_FromPolynomial(poly_t(rat_t(1))); }
    poly_t result(poly);

    for (long i = 1; i < power; ++i) { result *= poly; }

    return PyPolynomial_FromPolynomial(std::move(result));
}
PyObject* polynomial_floordiv(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_rem(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_rem(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_pow(PyObject* self, PyObject* other, PyObject*)
{
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_floordiv(PyObject* self, PyObject* other)
{
    Py_RETURN_NOTIMPLEMENTED;
}

PyObject* polynomial_neg(PyObject* self)
{
    return PyPolynomial_FromPolynomial(-cast_poly(self));
}
PyObject* polynomial_pos(PyObject* self)
{
    return PyPolynomial_FromPolynomial(poly_t(cast_poly(self)));
}

PyObject* polynomial_add(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        const auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        return PyPolynomial_FromPolynomial(lhs + rhs);
    }

    if (lhs_poly) {
        const auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            return PyPolynomial_FromPolynomial(poly + poly_t(mon, rat_t(1)));
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            return PyPolynomial_FromPolynomial(poly + poly_t(scalar));
        }
    } else {
        const auto& poly = cast_poly(other);

        if (is_monomial(self)) {
            const auto& mon = cast_mon(self);
            return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) + poly);
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, self)) {
            return PyPolynomial_FromPolynomial(poly_t(scalar) + poly);
        }
    }
    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_sub(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        const auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        return PyPolynomial_FromPolynomial(lhs - rhs);
    }

    if (lhs_poly) {
        const auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            return PyPolynomial_FromPolynomial(poly - poly_t(mon, rat_t(1)));
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            return PyPolynomial_FromPolynomial(poly - poly_t(scalar));
        }
    } else {
        const auto& poly = cast_poly(other);

        if (is_monomial(self)) {
            const auto& mon = cast_mon(self);
            return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) - poly);
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, self)) {
            return PyPolynomial_FromPolynomial(poly_t(scalar) - poly);
        }
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_div(PyObject* self, PyObject* other)
{
    // Only division by a scalar (rational) is supported for polynomials.
    if (!is_polynomial(self) || is_polynomial(other) || is_monomial(other)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const auto& poly = cast_poly(self);
    rat_t scalar;
    if (py_scalar_to_rat(scalar, other)) {
        if (scalar == rat_t(0)) {
            PyErr_SetString(PyExc_ArithmeticError, "division by zero");
            return nullptr;
        }

        return PyPolynomial_FromPolynomial(poly / scalar);
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_add(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        lhs += rhs;
        return self;
    }

    if (lhs_poly) {
        auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            poly += poly_t(mon, rat_t(1));
            return self;
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            poly += poly_t(scalar);
            return self;
        }

        Py_RETURN_NOTIMPLEMENTED;
    }

    const auto& poly = cast_poly(other);

    if (is_monomial(self)) {
        const auto& mon = cast_mon(self);
        return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) + poly);
    }

    rat_t scalar;
    if (py_scalar_to_rat(scalar, self)) {
        return PyPolynomial_FromPolynomial(poly_t(scalar) + poly);
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_sub(PyObject* self, PyObject* other)
{
    bool lhs_poly = is_polynomial(self);
    bool rhs_poly = is_polynomial(other);

    if (lhs_poly && rhs_poly) {
        auto& lhs = cast_poly(self);
        const auto& rhs = cast_poly(other);
        lhs -= rhs;
        return self;
    }

    if (lhs_poly) {
        auto& poly = cast_poly(self);
        if (is_monomial(other)) {
            const auto& mon = cast_mon(other);
            poly -= poly_t(mon, rat_t(1));
            return self;
        }

        rat_t scalar;
        if (py_scalar_to_rat(scalar, other)) {
            poly -= poly_t(scalar);
            return self;
        }

        Py_RETURN_NOTIMPLEMENTED;
    }

    const auto& poly = cast_poly(other);

    if (is_monomial(self)) {
        const auto& mon = cast_mon(self);
        return PyPolynomial_FromPolynomial(poly_t(mon, rat_t(1)) - poly);
    }

    rat_t scalar;
    if (py_scalar_to_rat(scalar, self)) {
        return PyPolynomial_FromPolynomial(poly_t(scalar) - poly);
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_inplace_div(PyObject* self, PyObject* other)
{
    if (!is_polynomial(self) || is_polynomial(other) || is_monomial(other)) {
        PyErr_SetString(
                PyExc_TypeError, "polynomials can only appear in the numerator"
        );
        return nullptr;
    }

    rat_t scalar;
    if (py_scalar_to_rat(scalar, other)) {
        if (scalar == rat_t(0)) {
            PyErr_SetString(PyExc_ArithmeticError, "division by zero");
            return nullptr;
        }

        cast_poly(self) /= scalar;
        return self;
    }

    Py_RETURN_NOTIMPLEMENTED;
}
PyObject* polynomial_str(PyObject* self)
{
    const auto& obj = cast_poly(self);
    std::stringstream ss;
    ss << obj;
    return PyUnicode_FromString(ss.str().c_str());
}
PyObject* polynomial_repr(PyObject* self)
{
    const auto& obj = cast_poly(self);
    std::stringstream ss;
    ss << obj;
    return PyUnicode_FromString(ss.str().c_str());
}
PyObject* polynomial_degree(PyObject* self)
{
    const auto& obj = cast_poly(self);
    return PyLong_FromLong(obj.degree());
}

PyObject* PyPolynomial_FromPolynomial(scalars::rational_poly_scalar&& poly
) noexcept
{
    auto* obj = reinterpret_cast<RPyPolynomial*>(
            RPyPolynomial_Type.tp_alloc(&RPyPolynomial_Type, 0)
    );

    if (obj == nullptr) { return nullptr; }

    try {
        construct_inplace(&obj->m_data, std::move(poly));
    } catch (std::exception& exc) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(obj);
}
