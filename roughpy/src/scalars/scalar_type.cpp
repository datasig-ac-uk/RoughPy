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

#include "scalar_type.h"

#include <unordered_map>

#include <boost/endian.hpp>

using namespace pybind11::literals;

using namespace rpy;

static PyMethodDef PyScalarMetaType_methods[] = {
        {nullptr, nullptr, 0, nullptr}
};

PyObject* PyScalarMetaType_call(PyObject*, PyObject*, PyObject*)
{
    // TODO: implement this?
    PyErr_SetString(PyExc_AssertionError, "doh");
    return nullptr;
}
static PyTypeObject PyScalarMetaType_type = {
        PyVarObject_HEAD_INIT(
                &PyType_Type, 0
        ) "_roughpy.ScalarMeta",          /* tp_name */
        sizeof(python::PyScalarMetaType), /* tp_basicsize */
        0,                                /* tp_itemsize */
        0,                                /* tp_dealloc */
        0,                                /* tp_vectorcall_offset */
        0,                                /* tp_getattr */
        0,                                /* tp_setattr */
        0,                                /* tp_as_async */
        0,                                /* tp_repr */
        0,                                /* tp_as_number */
        0,                                /* tp_as_sequence */
        0,                                /* tp_as_mapping */
        0,                                /* tp_hash */
        PyScalarMetaType_call,            /* tp_call */
        0,                                /* tp_str */
        0,                                /* tp_getattro */
        0,                                /* tp_setattro */
        0,                                /* tp_as_buffer */
        Py_TPFLAGS_TYPE_SUBCLASS,         /* tp_flags */
        PyDoc_STR("Scalar meta class"),   /* tp_doc */
        0,                                /* tp_traverse */
        0,                                /* tp_clear */
        0,                                /* tp_richcompare */
        0,                                /* tp_weaklistoffset */
        0,                                /* tp_iter */
        0,                                /* tp_iternext */
        PyScalarMetaType_methods,         /* tp_methods */
        0,                                /* tp_members */
        0,                                /* tp_getset */
        0,                                /* tp_base */
        0,                                /* tp_dict */
        0,                                /* tp_descr_get */
        0,                                /* tp_descr_set */
        0,                                /* tp_dictoffset */
        0,                                /* tp_init */
        0,                                /* tp_alloc */
        NULL,                             /* tp_new */
};

static PyMethodDef PyScalarTypeBase_methods[] = {
        {nullptr, nullptr, 0, nullptr}
};

static PyTypeObject PyScalarTypeBase_type = {
        PyVarObject_HEAD_INIT(NULL, 0) "_roughpy.ScalarTypeBase", /* tp_name */
        sizeof(python::PyScalarTypeBase), /* tp_basicsize */
        0,                                /* tp_itemsize */
        0,                                /* tp_dealloc */
        0,                                /* tp_vectorcall_offset */
        0,                                /* tp_getattr */
        0,                                /* tp_setattr */
        0,                                /* tp_as_async */
        0,                                /* tp_repr */
        0,                                /* tp_as_number */
        0,                                /* tp_as_sequence */
        0,                                /* tp_as_mapping */
        0,                                /* tp_hash */
        0,                                /* tp_call */
        0,                                /* tp_str */
        0,                                /* tp_getattro */
        0,                                /* tp_setattro */
        0,                                /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE
#if PY_VERSION_HEX >= 0x030A0000
                | Py_TPFLAGS_DISALLOW_INSTANTIATION
#endif
        ,                                        /* tp_flags */
        PyDoc_STR("Base class for scalar type"), /* tp_doc */
        0,                                       /* tp_traverse */
        0,                                       /* tp_clear */
        0,                                       /* tp_richcompare */
        0,                                       /* tp_weaklistoffset */
        0,                                       /* tp_iter */
        0,                                       /* tp_iternext */
        PyScalarTypeBase_methods,                /* tp_methods */
        0,                                       /* tp_members */
        0,                                       /* tp_getset */
        0,                                       /* tp_base */
        0,                                       /* tp_dict */
        0,                                       /* tp_descr_get */
        0,                                       /* tp_descr_set */
        0,                                       /* tp_dictoffset */
        0,                                       /* tp_init */
        0,                                       /* tp_alloc */
        0,                                       /* tp_new */
};

pybind11::handle python::get_scalar_metaclass()
{
    RPY_CHECK(PyType_Ready(&PyScalarMetaType_type) == 0);
    return py::handle(reinterpret_cast<PyObject*>(&PyScalarMetaType_type));
}
pybind11::handle python::get_scalar_baseclass()
{
    RPY_CHECK(PyType_Ready(&PyScalarTypeBase_type) == 0);
    return pybind11::handle(reinterpret_cast<PyObject*>(&PyScalarTypeBase_type)
    );
}
void python::PyScalarMetaType_dealloc(PyObject* arg)
{
    PyTypeObject* tp = Py_TYPE(arg);
    PyMem_Free(reinterpret_cast<PyScalarMetaType*>(arg)->ht_name);

    tp->tp_free(arg);
    Py_DECREF(tp);
}

static std::unordered_map<const scalars::ScalarType*, py::object>
        ctype_type_cache;

void python::register_scalar_type(
        const scalars::ScalarType* ctype, pybind11::handle py_type
)
{
    auto& found = ctype_type_cache[ctype];
    if (static_cast<bool>(found)) {
        RPY_THROW(std::runtime_error, "ctype already registered");
    }

    found = py::reinterpret_borrow<py::object>(py_type);
}

py::object python::to_ctype_type(const scalars::ScalarType* type)
{
    // The GIL must be held because we're working with Python objects anyway.
    if (type == nullptr) { RPY_THROW(std::runtime_error, "no matching ctype"); }
    const auto found = ctype_type_cache.find(type);
    if (found != ctype_type_cache.end()) { return found->second; }

    RPY_THROW(
            std::runtime_error,
            "no matching ctype for type " + type->info().name
    );
}

char python::format_to_type_char(const string& fmt)
{

    char python_format = 0;
    for (const auto& chr : fmt) {
        switch (chr) {
            case '<':// little-endian
#if BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_BIG_WORD
                RPY_THROW(std::runtime_error,
                        "non-native byte ordering not supported"
                );
#else
                break;
#endif
            case '>':// big-endian
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                RPY_THROW(std::runtime_error, 
                        "non-native byte ordering not supported"
                );
#else
                break;
#endif
            case '@':// native
            case '=':// native
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                break;
#endif
            case '!':// network ( = big-endian )
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                RPY_THROW(std::runtime_error,
                        "non-native byte ordering not supported"
                );
#else
                break;
#endif
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': break;
            default: python_format = chr; goto after_loop;
        }
    }
after_loop:
    return python_format;
}
string python::py_buffer_to_type_id(const py::buffer_info& info)
{
    using scalars::type_id_of;

    auto python_format = format_to_type_char(info.format);
    string format;
    switch (python_format) {
        case 'd': format = type_id_of<double>(); break;
        case 'f': format = type_id_of<float>(); break;
        case 'l': {
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<int>();
            } else {
                format = type_id_of<long long>();
            }
            break;
        }
        case 'q': format = scalars::type_id_of<long long>(); break;
        case 'L':
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<unsigned int>();
            } else {
                format = type_id_of<unsigned long long>();
            }
            break;
        case 'Q': format = type_id_of<unsigned long long>(); break;
        case 'i': format = type_id_of<int>(); break;
        case 'I': format = type_id_of<unsigned int>(); break;
        case 'n':
            format = type_id_of<scalars::signed_size_type_marker>();
            break;
        case 'N':
            format = type_id_of<scalars::unsigned_size_type_marker>();
            break;
        case 'h': format = type_id_of<short>(); break;
        case 'H': format = type_id_of<unsigned short>(); break;
        case 'b':
        case 'c': format = type_id_of<char>(); break;
        case 'B': format = type_id_of<unsigned char>(); break;
        default: RPY_THROW(std::runtime_error, "Unrecognised data format");
    }

    return format;
}

const scalars::ScalarType*
python::py_buffer_to_scalar_type(const py::buffer_info& info)
{
    using scalars::ScalarType;

    auto python_format = format_to_type_char(info.format);

    switch (python_format) {
        case 'f': return ScalarType::of<float>();
        case 'd': return ScalarType::of<double>();
        default:
            RPY_THROW(py::type_error,
                    "no matching type for buffer type " + string(&python_format)
            );
    }
    // TODO: Add custom type handling

    return ScalarType::of<double>();
}
const scalars::ScalarType* python::py_type_to_scalar_type(const py::type& type)
{
    if (type.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {
        return scalars::ScalarType::of<double>();
    } else if (type.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {
        return scalars::ScalarType::of<double>();
    }

    RPY_THROW(py::type_error,
            "no matching scalar type for type " + pytype_name(type)
    );
}

const scalars::ScalarType* python::py_arg_to_ctype(const pybind11::object& arg)
{
    if (py::isinstance(arg, python::get_scalar_metaclass())) {
        return reinterpret_cast<python::PyScalarMetaType*>(arg.ptr())->tp_ctype;
    }
    if (py::isinstance<py::str>(arg)) {
        return scalars::ScalarType::for_id(arg.cast<string>());
    }
    return nullptr;
}

py::type python::scalar_type_to_py_type(const scalars::ScalarType* type)
{
    if (type == scalars::ScalarType::of<float>()
        || type == scalars::ScalarType::of<double>()) {
        return py::reinterpret_borrow<py::type>(
                reinterpret_cast<PyObject*>(&PyFloat_Type)
        );
    }

    RPY_THROW(py::type_error, "no matching type for type " + type->info().name);
}

void python::init_scalar_types(pybind11::module_& m)
{

    PyScalarMetaType_type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyScalarMetaType_type) < 0) {
        throw py::error_already_set();
    }

    m.add_object(
            "ScalarMeta", reinterpret_cast<PyObject*>(&PyScalarMetaType_type)
    );

    Py_INCREF(&PyScalarMetaType_type);
#if PY_VERSION_HEX >= 0x03090000
    Py_SET_TYPE(&PyScalarTypeBase_type, &PyScalarMetaType_type);
#else
    reinterpret_cast<PyObject*>(&PyScalarTypeBase_type)->ob_type
            = &PyScalarMetaType_type;
#endif
    if (PyType_Ready(&PyScalarTypeBase_type) < 0) {
        pybind11::pybind11_fail(pybind11::detail::error_string());
    }

    m.add_object("ScalarTypeBase", get_scalar_baseclass());

    make_scalar_type(m, scalars::ScalarType::of<float>());
    make_scalar_type(m, scalars::ScalarType::of<double>());
    make_scalar_type(
            m, scalars::ScalarType::of<scalars::rational_scalar_type>()
    );
    make_scalar_type(m, scalars::ScalarType::of<scalars::half>());
    make_scalar_type(m, scalars::ScalarType::of<scalars::bfloat16>());
    make_scalar_type(
            m, scalars::ScalarType::of<scalars::rational_poly_scalar>()
    );
}
