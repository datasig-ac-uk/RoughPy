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

#ifndef RPY_PY_SCALARS_SCALAR_TYPE_H_
#define RPY_PY_SCALARS_SCALAR_TYPE_H_

#include "roughpy_module.h"

#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

namespace rpy {
namespace python {

struct PyScalarMetaType {
    PyHeapTypeObject tp_hto;
    char* ht_name;
    const scalars::ScalarType* tp_ctype;
};

struct PyScalarTypeBase {
    PyObject_VAR_HEAD
};

namespace dtl {

struct new_scalar_type_temps_manager {
    char* ht_tpname = nullptr;
    char* ht_name = nullptr;
    char* tp_doc = nullptr;
    PyScalarMetaType* cls = nullptr;

    ~new_scalar_type_temps_manager()
    {
        if (PyErr_Occurred()) { Py_CLEAR(cls); }

        PyMem_Free(ht_name);
        PyMem_Free(ht_tpname);
        PyMem_Free(tp_doc);
    }
};

}// namespace dtl

py::handle get_scalar_metaclass();
py::handle get_scalar_baseclass();

extern "C" void PyScalarMetaType_dealloc(PyObject* arg);

void register_scalar_type(const scalars::ScalarType* ctype, py::handle py_type);
py::object to_ctype_type(const scalars::ScalarType* type);

inline void make_scalar_type(py::module_& m, const scalars::ScalarType* ctype)
{
    py::object mcs = py::reinterpret_borrow<py::object>(get_scalar_metaclass());

    py::handle base = get_scalar_baseclass();
    //    py::handle bases(PyTuple_Pack(1, base.ptr()));
    //    py::handle base((PyObject*) &PyBaseObject_Type);

    auto* mcs_tp = reinterpret_cast<PyTypeObject*>(mcs.ptr());

    const auto& name = ctype->info().name;
    py::str ht_name(name);
    dtl::new_scalar_type_temps_manager tmp_manager;

#if PY_VERSION_HEX >= 0x03090000
    py::object ht_module(m);
#endif

    dimn_t no_letters = name.size() + 1;
    tmp_manager.ht_name = reinterpret_cast<char*>(PyMem_Malloc(no_letters));
    if (tmp_manager.ht_name == nullptr) {
        PyErr_NoMemory();
        throw py::error_already_set();
    }
    memcpy(tmp_manager.ht_name, name.c_str(), no_letters);

    tmp_manager.cls
            = reinterpret_cast<PyScalarMetaType*>(mcs_tp->tp_alloc(mcs_tp, 0));
    if (tmp_manager.cls == nullptr) { throw py::error_already_set(); }
    auto* hto = reinterpret_cast<PyHeapTypeObject*>(&tmp_manager.cls->tp_hto);
    auto* type = &hto->ht_type;

    type->tp_flags
            = (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE
#if PY_VERSION_HEX >= 0x030A0000
               | Py_TPFLAGS_DISALLOW_INSTANTIATION
#endif
            );
#if PY_VERSION_HEX >= 0x03090000
    hto->ht_module = ht_module.release().ptr();
#endif

    type->tp_as_async = &hto->as_async;
    type->tp_as_buffer = &hto->as_buffer;
    type->tp_as_sequence = &hto->as_sequence;
    type->tp_as_mapping = &hto->as_mapping;
    type->tp_as_number = &hto->as_number;

    type->tp_base = reinterpret_cast<PyTypeObject*>(base.inc_ref().ptr());
    //    type->tp_bases = bases.ptr();

    type->tp_doc = tmp_manager.tp_doc;

    hto->ht_qualname = ht_name.release().ptr();
    hto->ht_name = hto->ht_qualname;

    type->tp_name = tmp_manager.ht_name;
    tmp_manager.ht_name = nullptr;

    type->tp_basicsize = sizeof(PyScalarMetaType);
    type->tp_itemsize = 0;

    type->tp_dealloc = PyScalarMetaType_dealloc;

    tmp_manager.cls->tp_ctype = ctype;

    if (PyType_Ready(type) < 0) {
        py::pybind11_fail("Error " + py::detail::error_string());
    }

    py::handle h_class(reinterpret_cast<PyObject*>(tmp_manager.cls));
    register_scalar_type(ctype, h_class);
    m.add_object(name.c_str(), h_class);
}

inline const scalars::ScalarType* to_stype_ptr(const py::handle& arg)
{
    if (!py::isinstance(arg, get_scalar_metaclass())) {
        RPY_THROW(py::type_error, "argument is not a valid scalar type");
    }
    return reinterpret_cast<PyScalarMetaType*>(arg.ptr())->tp_ctype;
}

char format_to_type_char(const string& fmt);
string py_buffer_to_type_id(const py::buffer_info& info);

const scalars::ScalarType* py_buffer_to_scalar_type(const py::buffer_info& info
);
const scalars::ScalarType* py_type_to_scalar_type(const py::type& type);
const scalars::ScalarType* py_arg_to_ctype(const py::object& arg);

py::type scalar_type_to_py_type(const scalars::ScalarType* type);

inline string pytype_name(const py::type& type)
{
    return {reinterpret_cast<PyTypeObject*>(type.ptr())->tp_name};
}

void init_scalar_types(py::module_& m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_SCALARS_SCALAR_TYPE_H_
