//
// Created by sam on 2/29/24.
//

#include "pytype_conversion.h"

#include "roughpy/core/check.h"           // for throw_exception, RPY_CHECK

#include "scalars/scalar_type.h"

using namespace rpy;
using namespace rpy::python;

bool python::is_imported_type(
        py::handle obj,
        const char* module,
        const char* type_name
)
{
    try {
        auto mod = py::module_::import(module);
        auto tp = mod.attr(type_name);
        return py::isinstance(obj, tp);
    } catch (py::import_error&) {
        return false;
    } catch (py::attribute_error&) {
        return false;
    }
}

static PyObject* p_scalar_type_mapping = nullptr;

py::object python::init_scalar_mapping()
{
    p_scalar_type_mapping = PyDict_New();
    return py::reinterpret_steal<py::object>(p_scalar_type_mapping);
}

bool python::is_py_scalar_type(py::handle obj)
{
    py::gil_scoped_acquire gil;
    auto* tp = Py_TYPE(obj.ptr());

    return static_cast<bool>(PyDict_Contains(
            p_scalar_type_mapping,
            reinterpret_cast<PyObject*>(tp)
    ));
}

const scalars::ScalarType* python::type_of_pyscalar(py::handle obj)
{
    auto* tp = reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()));
    return py_type_to_type(tp);
}

const scalars::ScalarType* python::py_type_to_type(py::handle obj) {
    py::gil_scoped_acquire gil;
    if (obj.ptr() == (PyObject*) &PyFloat_Type) {
        return *scalars::scalar_type_of<double>();
    }
    if (obj.ptr() == (PyObject*) &PyLong_Type) {
        return *scalars::scalar_type_of<double>();
    }

    if (PyDict_Contains(p_scalar_type_mapping, obj.ptr()) == 0) {
        RPY_THROW(py::type_error, "unrecognised scalar type");
    }

    auto* info = PyDict_GetItem(p_scalar_type_mapping, obj.ptr());

    RPY_CHECK(Py_TYPE(info) == &PyTuple_Type);
    RPY_CHECK(PyTuple_GET_SIZE(info) == 2);
    // pair (scalar type, converter)

    return to_stype_ptr({PyTuple_GetItem(info, 0)});
}

devices::TypeInfo python::py_type_to_type_info(py::handle pytype)
{
    return py_type_to_type(pytype)->type_info();
}
