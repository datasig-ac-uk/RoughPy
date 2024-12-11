//
// Created by sam on 11/12/24.
//

#include "signature_arguments.h"

#ifdef ROUGHPY_WITH_NUMPY
#  define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#  include <numpy/arrayobject.h>
#endif


#include "roughpy/core/construct_inplace.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/archives.h"
#include "roughpy/platform/errors.h"

#include "algebra/context.h"
#include "args/kwargs_to_path_metadata.h"
#include "args/numpy.h"
#include "intervals/interval.h"
#include "scalars/scalar_type.h"


using namespace rpy;
using namespace rpy::python;


static int resolution_converter(PyObject* object, void* out)
{
    if (Py_TYPE(object) == &PyFloat_Type) {
        *reinterpret_cast<optional<resolution_t>*>(out)
                = python::param_to_resolution(PyFloat_AsDouble(object));
    } else if (Py_TYPE(object) == &PyLong_Type) {
        *reinterpret_cast<optional<resolution_t>*>(out) = PyLong_AsLong(object);
#if PY_VERSION_HEX >= 0x030A0000
    } else if (Py_IsNone(object)) {
#else
    } else if (object == Py_None) {
#endif
        // do nothing, use default
    } else {
        PyErr_SetString(
                PyExc_TypeError,
                "resolution should be either float or int"
        );
        return 0;
    }
    return 1;
}



int python::parse_sig_args(
        PyObject* args,
        PyObject* kwargs,
        const streams::StreamMetadata* smeta,
        SigArgs* sigargs
)
{
    static const char* kwords[]
            = {"interval_or_inf",
               "sup",
               "resolution",
               "ctx",
               "depth",
               "dtype",
               nullptr};

    PyObject* interval_or_inf = nullptr;
    PyObject* ctx = nullptr;
    PyObject* py_sup = nullptr;
    PyObject* dtype = nullptr;
    deg_t depth = 0;

    auto result = PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|OOO&$O!iO",
            const_cast<char**>(kwords),
            &interval_or_inf,
            &py_sup,
            &resolution_converter,
            &sigargs->resolution,
            &python::RPyContext_Type,
            &ctx,
            &depth,
            &dtype
    );
    if (result == 0) { return -1; }

    // First decide if we're given an interval, inf/sup pair, or global
    if (interval_or_inf == nullptr || interval_or_inf == Py_None) {
        // Global, nothing to do as optional default is empty.
    } else if (Py_TYPE(interval_or_inf) == &PyFloat_Type || Py_TYPE(interval_or_inf) == &PyLong_Type) {
        if (py_sup == nullptr) {
            // In this case, we're expecting the first argument to be resolution
            if (Py_TYPE(interval_or_inf) == &PyFloat_Type) {
                PyErr_SetString(
                        PyExc_TypeError,
                        "expecting integer resolution"
                );
                return -1;
            }
            sigargs->resolution
                    = static_cast<resolution_t>(PyLong_AsLong(interval_or_inf));

            sigargs->interval = {};
        } else {
            param_t inf;
            param_t sup;

            if (Py_TYPE(interval_or_inf) == &PyFloat_Type) {
                inf = PyFloat_AsDouble(interval_or_inf);
            } else {
                inf = PyLong_AsDouble(interval_or_inf);
            }

            if (Py_TYPE(py_sup) == &PyFloat_Type) {
                sup = PyFloat_AsDouble(py_sup);
            } else if (Py_TYPE(py_sup) == &PyLong_Type) {
                sup = PyLong_AsDouble(py_sup);
            } else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "expected float value for sup"
                );
                return -1;
            }

            if (inf > sup) {
                PyErr_SetString(
                        PyExc_ValueError,
                        "inf must not be larger than sup"
                );
                return -1;
            }
            sigargs->interval = intervals::RealInterval(inf, sup);
        }
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::RealInterval>()
                          .ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = py::handle(interval_or_inf)
                                    .cast<const intervals::RealInterval&>();
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::DyadicInterval>()
                          .ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = intervals::RealInterval(
                py::handle(interval_or_inf)
                        .cast<const intervals::DyadicInterval&>()
        );
    } else if (Py_TYPE(interval_or_inf)
               == (PyTypeObject*) py::type::of<intervals::Interval>().ptr()) {
        if (py_sup != nullptr) {
            if (resolution_converter(py_sup, &sigargs->resolution) == 0) {
                return -1;
            }
        }

        sigargs->interval = intervals::RealInterval(
                py::handle(interval_or_inf).cast<const intervals::Interval&>()
        );
    } else {
        PyErr_SetString(
                PyExc_TypeError,
                "unexpected type for argument interval_or_inf"
        );
        return -1;
    }

    // Finally we need to decide if a change of context is necessary.
    if (ctx == nullptr) {
        if (depth == 0 && dtype == nullptr) {
            sigargs->ctx = smeta->default_context;
            return 0;
        }

        const scalars::ScalarType* ctype = smeta->data_scalar_type;
        if (dtype != nullptr) {
            if (Py_TYPE(dtype)
                == (PyTypeObject*) python::get_scalar_metaclass().ptr()) {
                ctype = ((python::PyScalarMetaType*) dtype)->tp_ctype;
            } else if (Py_TYPE(dtype) == &PyUnicode_Type) {
                const auto* dtype_str = PyUnicode_AsUTF8(dtype);
                const auto ctype_o = scalars::get_type(dtype_str);
                if (ctype_o) {
                    PyErr_SetString(
                            PyExc_TypeError,
                            "unrecognised scalar type id"
                    );
                    return -1;
                }
                ctype = *ctype_o;
            }
#ifdef ROUGHPY_WITH_NUMPY
            else if (Py_TYPE(dtype) == &PyArrayDescr_Type) {
                ctype = python::npy_dtype_to_ctype(
                        py::reinterpret_borrow<py::dtype>(dtype)
                );
                if (ctype == nullptr) {
                    PyErr_SetString(
                            PyExc_ValueError,
                            "unsupported scalar type"
                    );
                    return -1;
                }
            }
#endif
            else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "unexpected argument type for argument dtype"
                );
                return -1;
            }
        }

        sigargs->ctx = smeta->default_context->get_alike(depth, ctype);
    } else {
        // a provided context always takes priority over other configurations.
        // This cast is infallible because we've already checked that the type
        // is correct.
        sigargs->ctx = python::ctx_cast(ctx);
    }

    return 0;
}

