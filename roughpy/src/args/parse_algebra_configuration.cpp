//
// Created by sam on 2/19/24.
//

#include "parse_algebra_configuration.h"

#include "algebra/context.h"
#include "numpy.h"
#include "scalars/scalar_type.h"

#include <roughpy/algebra/context.h>

using namespace rpy;
using namespace rpy::python;


AlgebraConfiguration python::parse_algebra_configuration(py::kwargs& kwargs)
{
    AlgebraConfiguration result{nullptr, {}, {}, nullptr};
    bool warned = false;

    static constexpr const char* ctx_kwords[] = {
        "ctx",
        "context"
    };
    const char* ctx_name = nullptr;

    for (const auto* name : ctx_kwords) {
        if (kwargs.contains(name)) {
            if (ctx_name == nullptr) {
                ctx_name = name;
                auto ctx = kwargs_pop(kwargs, name);
                result.ctx = ctx_cast(ctx.ptr());
            } else {
                // We need to pop repeated arguments, because otherwise they
                // will trigger an additional warning
                auto _ = kwargs_pop(kwargs, name);
                PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                                 "two keyword arguments specifying context "
                                 "have been provided, \"%s\" will be ignored",
                                 name);
            }
        }
    }

    static constexpr const char* width_kword = "width";

    if (kwargs.contains(width_kword)) {
        result.width = kwargs_pop(kwargs, width_kword).cast<deg_t>();
        if (result.ctx) {
            PyErr_WarnFormat(PyExc_RuntimeWarning,
                             1,
                             "providing both \"%s\" and \"width\" "
                             "ignores the \"width\" parameter. "
                             "Note that the \"width\" parameter is deprecated,"
                             "and you should use the \"ctx\" argument instead.",
                             ctx_name);
            warned = true;
        }
        if (!warned) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "using the \"width\" keyword argument is deprecated, "
                         "instead you should use a context and the \"ctx\""
                         " keyword argument to specify the "
                         "algebra configuration",
                         1);
            warned = true;
        }
    }

    static constexpr const char* depth_keyword = "depth";
    if (kwargs.contains(depth_keyword)) {
        result.depth = kwargs_pop(kwargs, depth_keyword).cast<deg_t>();
        if (!warned && result.ctx) {
            PyErr_WarnFormat(PyExc_RuntimeWarning,
                             1,
                             "providing both \"%s\" and \"depth\" "
                             "ignores the \"depth\" parameter. "
                             "Note that the \"depth\" parameter is deprecated,"
                             "and you should use the \"ctx\" argument instead.",
                             ctx_name);
            warned = true;
        }
        if (!warned) {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "using the \"depth\" keyword argument is deprecated, "
                         "instead you should use a context and the \"ctx\""
                         " keyword argument to specify the "
                         "algebra configuration",
                         1);
            warned = true;
        }
    }

    static constexpr const char* coeff_kwords[] = {"dtype", "ctype", "coeffs"};
    for (const auto* name : coeff_kwords) {
        if (kwargs.contains(name)) {
            if (result.scalar_type == nullptr) {
                const auto obj = kwargs_pop(kwargs, name);
#ifdef ROUGHPY_WITH_NUMPY
                if (py::isinstance<py::dtype>(obj)) {
                    result.scalar_type = npy_dtype_to_ctype(obj);
                } else {
                    result.scalar_type = py_arg_to_ctype(obj);
                }
#else
                result.scalar_type = py_arg_to_ctype(obj);
#endif
            } else {
                auto _ = kwargs_pop(kwargs, name);
                PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                                 "multiple keyword arguments specifying scalar "
                                 "type have been provided, \"%s\" will be "
                                 "ignored",
                                 name);
            }

            if (!warned && result.ctx) {
                PyErr_WarnFormat(PyExc_RuntimeWarning,
                                 1,
                                 "providing both \"%s\" and \"%s\" "
                                 "ignores the \"%s\" parameter. "
                                 "Note that the \"%s\" parameter is deprecated,"
                                 "and you should use the \"ctx\" argument "
                                 "instead.",
                                 ctx_name, name, name, name
                );
                warned = true;
            }
            if (!warned) {
                PyErr_WarnFormat(PyExc_DeprecationWarning,
                                 1,
                                 "using the \"%s\" keyword argument is"
                                 " deprecated, instead you should use a context"
                                 " and the \"ctx\" keyword argument to specify "
                                 "the algebra configuration",
                                 name);
                warned = true;
            }
        }
    }

    return result;
}
