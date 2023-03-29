#include "context.h"

#include <pybind11/stl.h>

#include "args/numpy.h"
#include "scalars/scalar_type.h"
#include "lie_key_iterator.h"
#include "tensor_key_iterator.h"


using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

#ifdef ROUGHPY_WITH_NUMPY
static FreeTensor context_compute_signature_numpy_darray(const python::PyContext &ctx,
                                                         const py::array_t<double, py::array::forcecast> &array) {
    assert(array.ndim() == 2);
    auto shape = array.shape();

    const auto n_increments = shape[0];
    const auto width = shape[1];

    assert(width == ctx->width());

    const auto* ctype = ctx->ctype();
    SignatureData request;
    request.data_stream.set_ctype(ctype);
    request.data_stream.set_elts_per_row(width);
    request.data_stream.reserve_size(n_increments);
    for (dimn_t i = 0; i < n_increments; ++i) {
        request.data_stream.push_back(scalars::ScalarPointer(ctype, array.data(i, 0)));
    }
    request.vector_type = VectorType::Dense;

    auto sig = ctx->signature(request);

    return sig;
    //    return ctx->signature(request);
}
#endif

static python::PyContext py_get_context(deg_t width, deg_t depth, const py::object &ctype, const py::kwargs &kwargs) {
    //TODO: Make this accept extra arguments.
    return get_context(width, depth, python::to_stype_ptr(ctype), {});
}

void python::init_context(py::module_ &m) {

    py::class_<PyContext> klass(m, "Context");

    klass.def_property_readonly("width", [](const PyContext &ctx) { return ctx->width(); });
    klass.def_property_readonly("depth", [](const PyContext &ctx) { return ctx->depth(); });
    klass.def_property_readonly("ctype", [](const PyContext &ctx) { return to_ctype_type(ctx->ctype()); });
    klass.def(
        "lie_size", [](const PyContext &ctx, deg_t degree) { return ctx->lie_size(degree); }, "degree"_a);
    klass.def(
        "tensor_size", [](const PyContext &ctx, deg_t degree) { return ctx->tensor_size(degree); }, "degree"_a);
    klass.def(
        "cbh", [](const PyContext &ctx, std::vector<Lie> lies, algebra::VectorType vtype) {
          return ctx->cbh(lies, vtype);
        },
        "lies"_a, "vec_type"_a);

#ifndef ESIG_NO_NUMPY
    klass.def("compute_signature", context_compute_signature_numpy_darray, "data"_a);
#endif
    klass.def(
        "to_logsignature", [](const PyContext &ctx, const FreeTensor &sig) {
          return ctx->tensor_to_lie(sig.log());
        },
        "signature"_a);

    klass.def("iterate_lie_keys", [](const PyContext &ctx) {
      return python::PyLieKeyIterator(ctx);
    });
    klass.def("iterate_tensor_keys", [](const PyContext &ctx) {
      return python::PyTensorKeyIterator(ctx->width(), ctx->depth());
    });

    klass.def("lie_to_tensor", [](const PyContext& ctx, const Lie& arg) {
            return ctx->lie_to_tensor(arg);
        }, "arg"_a);
    klass.def("tensor_to_lie", [](const PyContext& ctx, const FreeTensor& arg) {
             return ctx->tensor_to_lie(arg);
         }, "arg"_a);


    klass.def("__enter__", [](const PyContext& ctx) { return ctx; });
    klass.def("__exit__", [](const PyContext& ctx, py::handle exc_type, py::handle exc_value, py::handle tb) {
             return py::none();
         }, "exc_type"_a, "exc_value"_a, "traceback"_a);

    m.def("get_context", py_get_context, "width"_a, "depth"_a, "coeffs"_a = py::none());

}
