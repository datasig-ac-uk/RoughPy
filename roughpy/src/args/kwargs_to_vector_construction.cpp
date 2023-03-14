#include "kwargs_to_vector_construction.h"

#include "algebra/context.h"
#include "scalars/scalar_type.h"

using namespace rpy;


python::PyVectorConstructionHelper python::kwargs_to_construction_data(const pybind11::kwargs &kwargs) {

    PyVectorConstructionHelper helper;

    if (kwargs.contains("context")) {
        helper.ctx = kwargs["context"].cast<PyContext>().to_context();
        helper.width = helper.ctx->width();
        helper.depth = helper.ctx->depth();
        helper.ctype = helper.ctx->ctype();
        helper.ctype_requested = true;
    } else {
        if (kwargs.contains("ctype")) {
            helper.ctype = py_arg_to_ctype(kwargs["ctype"]);
            helper.ctype_requested = true;
        }

        if (kwargs.contains("depth")) {
            helper.depth = kwargs["depth"].cast<deg_t>();
        } else {
            helper.depth = 2;
        }

        if (kwargs.contains("width")) {
            helper.width = kwargs["width"].cast<deg_t>();
        }
    }

    if (kwargs.contains("vector_type")) {
        helper.vtype = kwargs["vector_type"].cast<algebra::VectorType>();
        helper.vtype_requested = true;
    }

    if (kwargs.contains("keys")) {
        const auto &arg = kwargs["keys"];
        if (py::isinstance<key_type>(arg)) {
        } else if (py::isinstance<py::buffer>(arg)) {
            auto key_info = arg.cast<py::buffer>().request();
        }
    }

    if (helper.width != 0 && helper.depth != 0 && helper.ctype != nullptr) {
        helper.ctx = algebra::get_context(helper.width, helper.depth, helper.ctype);
    }

    return helper;
}
