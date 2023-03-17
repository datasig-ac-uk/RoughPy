#include "scalar_type.h"

#include <unordered_map>

#include <boost/endian.hpp>

using namespace pybind11::literals;

using namespace rpy;

static PyMethodDef PyScalarMetaType_methods[] = {
    {nullptr, nullptr, 0, nullptr}};

PyObject *PyScalarMetaType_call(PyObject *, PyObject *, PyObject *) {
    PyErr_SetString(PyExc_AssertionError, "doh");
    return nullptr;
}

static PyTypeObject PyScalarMetaType_type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
        .tp_name = "_roughpy.ScalarMeta",
    .tp_basicsize = sizeof(python::PyScalarMetaType),
    .tp_itemsize = 0,
    .tp_call = PyScalarMetaType_call,
    .tp_flags = Py_TPFLAGS_TYPE_SUBCLASS,
    .tp_doc = PyDoc_STR("Scalar meta class"),
    .tp_methods = PyScalarMetaType_methods};

static PyMethodDef PyScalarTypeBase_methods[] = {
    {nullptr, nullptr, 0, nullptr}};

static PyTypeObject PyScalarTypeBase_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
        .tp_name = "_roughpy.ScalarTypeBase",
    .tp_basicsize = sizeof(python::PyScalarTypeBase),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .tp_doc = PyDoc_STR("Base class for scalar type"),
    .tp_methods = PyScalarTypeBase_methods};

pybind11::handle python::get_scalar_metaclass() {
    assert(PyType_Ready(&PyScalarMetaType_type) == 0);
    return py::handle(reinterpret_cast<PyObject *>(&PyScalarMetaType_type));
}
pybind11::handle python::get_scalar_baseclass() {
    assert(PyType_Ready(&PyScalarTypeBase_type) == 0);
    return pybind11::handle(reinterpret_cast<PyObject *>(&PyScalarTypeBase_type));
}
void python::PyScalarMetaType_dealloc(PyObject *arg) {
    PyTypeObject *tp = Py_TYPE(arg);
    PyMem_Free(reinterpret_cast<PyScalarMetaType *>(arg)->ht_name);

    tp->tp_free(arg);
    Py_DECREF(tp);
}

static std::unordered_map<const scalars::ScalarType *, py::object> ctype_type_cache;

void python::register_scalar_type(const scalars::ScalarType *ctype, pybind11::handle py_type) {
    auto &found = ctype_type_cache[ctype];
    if (static_cast<bool>(found)) {
        throw std::runtime_error("ctype already registered");
    }

    found = py::reinterpret_borrow<py::object>(py_type);
}

py::object python::to_ctype_type(const scalars::ScalarType *type) {
    // The GIL must be held because we're working with Python objects anyway.
    if (type == nullptr) {
        throw std::runtime_error("no matching ctype");
    }
    const auto found = ctype_type_cache.find(type);
    if (found != ctype_type_cache.end()) {
        return found->second;
    }

    throw std::runtime_error("no matching ctype for type " + type->info().name);
}

char python::format_to_type_char(const std::string &fmt) {

    char python_format = 0;
    for (const auto &chr : fmt) {
        switch (chr) {
            case '<':// little-endian
#if BOOST_ENDIAN_BIG_BYTE || BOOST_ENDIAN_BIG_WORD
                throw std::runtime_error("non-native byte ordering not supported");
#else
                break;
#endif
            case '>':// big-endian
#if BOOST_ENDIAN_LITTLE_BYTE || BOOST_ENDIAN_LITTLE_WORD
                throw std::runtime_error("non-native byte ordering not supported");
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
                throw std::runtime_error("non-native byte ordering not supported");
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
            case '9':
                break;
            default:
                python_format = chr;
                goto after_loop;
        }
    }
after_loop:
    return python_format;
}
std::string python::py_buffer_to_type_id(const py::buffer_info &info) {
    using scalars::type_id_of;

    auto python_format = format_to_type_char(info.format);
    std::string format;
    switch (python_format) {
        case 'd':
            format = type_id_of<double>();
            break;
        case 'f':
            format = type_id_of<float>();
            break;
        case 'l': {
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<int>();
            } else {
                format = type_id_of<long long>();
            }
            break;
        }
        case 'q':
            format = scalars::type_id_of<long long>();
            break;
        case 'L':
            if (info.itemsize == sizeof(int)) {
                format = type_id_of<unsigned int>();
            } else {
                format = type_id_of<unsigned long long>();
            }
            break;
        case 'Q':
            format = type_id_of<unsigned long long>();
            break;
        case 'i':
            format = type_id_of<int>();
            break;
        case 'I':
            format = type_id_of<unsigned int>();
            break;
        case 'n':
            format = type_id_of<scalars::signed_size_type_marker>();
            break;
        case 'N':
            format = type_id_of<scalars::unsigned_size_type_marker>();
            break;
        case 'h':
            format = type_id_of<short>();
            break;
        case 'H':
            format = type_id_of<unsigned short>();
            break;
        case 'b':
        case 'c':
            format = type_id_of<char>();
            break;
        case 'B':
            format = type_id_of<unsigned char>();
            break;
        default:
            throw std::runtime_error("Unrecognised data format");
    }

    return format;
}

const scalars::ScalarType *python::py_buffer_to_scalar_type(const py::buffer_info &info) {
    using scalars::ScalarType;

    auto python_format = format_to_type_char(info.format);

    switch (python_format) {
        case 'f':
            return ScalarType::of<float>();
        case 'd':
            return ScalarType::of<double>();
        default:
            throw py::type_error("no matching type for buffer type " + std::string(&python_format));
    }
    // TODO: Add custom type handling

    return ScalarType::of<double>();
}
const scalars::ScalarType *python::py_type_to_scalar_type(const py::type &type) {
    if (Py_Is(type.ptr(), reinterpret_cast<PyObject *>(&PyFloat_Type))) {
        return scalars::ScalarType::of<double>();
    } else if (Py_Is(type.ptr(), reinterpret_cast<PyObject *>(&PyLong_Type))) {
        return scalars::ScalarType::of<double>();
    }

    throw py::type_error("no matching scalar type for type " + pytype_name(type));
}

const scalars::ScalarType *python::py_arg_to_ctype(const pybind11::object &arg) {
    if (py::isinstance(arg, python::get_scalar_metaclass())) {
        return reinterpret_cast<python::PyScalarMetaType *>(arg.ptr())->tp_ctype;
    }
    if (py::isinstance<py::str>(arg)) {
        return scalars::ScalarType::for_id(arg.cast<std::string>());
    }
    return nullptr;
}

py::type python::scalar_type_to_py_type(const scalars::ScalarType *type) {
    if (type == scalars::ScalarType::of<float>() || type == scalars::ScalarType::of<double>()) {
        return py::reinterpret_borrow<py::type>(reinterpret_cast<PyObject *>(&PyFloat_Type));
    }

    throw py::type_error("no matching type for type " + type->info().name);
}

void python::init_scalar_types(pybind11::module_ &m) {

    PyScalarMetaType_type.tp_base = &PyType_Type;
    if (PyType_Ready(&PyScalarMetaType_type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("ScalarMeta", reinterpret_cast<PyObject *>(&PyScalarMetaType_type));

    Py_INCREF(&PyScalarMetaType_type);
    Py_SET_TYPE(&PyScalarTypeBase_type, &PyScalarMetaType_type);
    if (PyType_Ready(&PyScalarTypeBase_type) < 0) {
        pybind11::pybind11_fail(pybind11::detail::error_string());
    }

    m.add_object("ScalarTypeBase", get_scalar_baseclass());

    make_scalar_type(m, scalars::ScalarType::of<float>());
    make_scalar_type(m, scalars::ScalarType::of<double>());
    make_scalar_type(m, scalars::ScalarType::of<scalars::rational_scalar_type>());
}
