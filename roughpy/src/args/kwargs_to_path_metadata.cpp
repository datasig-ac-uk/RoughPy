#include "kwargs_to_path_metadata.h"

#include "algebra/context.h"
#include "scalars/scalar_type.h"

#include "numpy.h"

using namespace rpy;

python::PyStreamMetaData python::kwargs_to_metadata(const pybind11::kwargs &kwargs) {

    PyStreamMetaData md{
        0,                                // width
        0,                                // depth
        intervals::RealInterval{0.0, 1.0},// support
        nullptr,                          // context
        nullptr,                          // scalar type
        algebra::VectorType::Dense,       // vector type
        3                                // default resolution
    };

    if (kwargs.contains("ctx")) {
        auto ctx = kwargs["ctx"];
        if (!py::isinstance(ctx, reinterpret_cast<PyObject*>(&RPyContext_Type))) {
            throw py::type_error("expected a Context object");
        }
        md.ctx = python::ctx_cast(ctx.ptr());
        md.width = md.ctx->width();
        md.scalar_type = md.ctx->ctype();
    } else {

        if (kwargs.contains("width")) {
            md.width = kwargs["width"].cast<rpy::deg_t>();
        }
        if (kwargs.contains("depth")) {
            md.depth = kwargs["depth"].cast<rpy::deg_t>();
        }
        if (kwargs.contains("ctype")) {
            md.scalar_type = python::py_arg_to_ctype(kwargs["ctype"]);
        }
#ifdef ROUGHPY_WITH_NUMPY
        else if (kwargs.contains("dtype")) {
            auto dtype = kwargs["dtype"];
            if (py::isinstance<py::dtype>(dtype)) {
                md.scalar_type = npy_dtype_to_ctype(dtype);
            } else {
                md.scalar_type = py_arg_to_ctype(dtype);
            }
        }
#endif
    }

    if (kwargs.contains("vtype")) {
        md.vector_type = kwargs["vtype"].cast<algebra::VectorType>();
    }

    return md;
}
