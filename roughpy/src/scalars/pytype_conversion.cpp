//
// Created by sam on 2/29/24.
//

#include "pytype_conversion.h"

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

const scalars::ScalarType* type_of_pyscalar(py::handle obj)
{
    py::gil_scoped_acquire gil;

    auto* tp = reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()));

    if (PyDict_Contains(p_scalar_type_mapping, tp) == 0) {
        RPY_THROW(py::type_error, "unrecognised scalar type");
    }

    auto* info = PyDict_GetItem(p_scalar_type_mapping, tp)

            RPY_CHECK(Py_TYPE(info) == PyTuple_Type);
    RPY_CHECK(PyTuple_GET_SIZE(info) == 2);
    // pair (scalar type, converter)

    return to_stype_ptr({PyTuple_GetItem(info, 0)});
}
