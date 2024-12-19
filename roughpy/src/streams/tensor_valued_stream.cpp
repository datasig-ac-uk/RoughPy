//
// Created by sammorley on 10/12/24.
//

#include "tensor_valued_stream.h"

#include <memory>

#include "roughpy/algebra/free_tensor.h"
#include "roughpy/streams/value_stream.h"

#include "stream.h"
#include "signature_arguments.h"

#include <args/kwargs_to_path_metadata.h>
#include <range/v3/view/move.hpp>
#include <roughpy/streams/lie_increment_stream.h>

using namespace pybind11::literals;
using namespace rpy;
using namespace rpy::streams;

using algebra::FreeTensor;


struct RPySimpleTensorValuedStream
{
    PyObject_VAR_HEAD//
    std::shared_ptr<const ValueStream<FreeTensor>> p_data;
};


py::object python::TensorValuedStream_FromPtr(
    std::shared_ptr<const ValueStream<FreeTensor>> ptr)
{
    auto new_obj = py::reinterpret_steal<py::object>(
        TensorValuedStream_Type.tp_alloc(&TensorValuedStream_Type, 0));

    if (new_obj) {
        auto* data = reinterpret_cast<RPySimpleTensorValuedStream*>(new_obj.
            ptr());
        construct_inplace(&data->p_data, std::move(ptr));
    }

    return new_obj;
}


extern "C" {
static PyObject* stvs_new(PyTypeObject* subtype,
                          PyObject* args,
                          PyObject* kwargs)
{
    static const char* kwlist[] = {
            "increment_stream", "initial_value", "domain", nullptr
    };

    python::RPyStream* incr_stream;
    PyObject* initial_value;
    PyObject* domain;

    auto* ft_type = reinterpret_cast<PyTypeObject*>(py::type::of<FreeTensor>().
        ptr());
    auto* interval_type = reinterpret_cast<PyTypeObject*>(py::type::of<
        intervals::RealInterval>().ptr());

    if (PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O!O!O!",
                                    const_cast<char**>(kwlist),
                                    &python::RPyStream_Type,
                                    &incr_stream,
                                    ft_type,
                                    &initial_value,
                                    interval_type,
                                    &domain
    ) == 0) {
        RPY_DBG_ASSERT(PyErr_Occurred() != nullptr);
        return nullptr;
    }

    auto new_obj = py::reinterpret_steal<py::object>(
        subtype->tp_alloc(subtype, 0));
    if (!new_obj) {
        RPY_DBG_ASSERT(PyErr_Occurred() != nullptr);
        return nullptr;
    }

    auto* self = reinterpret_cast<RPySimpleTensorValuedStream*>(new_obj.ptr());

    auto success = python::with_caught_exceptions([&]() {
        // Make sure the shared pointer is properly initialised
        construct_inplace(&self->p_data,
                          make_simple_tensor_valued_stream(
                              incr_stream->m_data.impl(),
                              py::cast<const FreeTensor&>(initial_value),
                              py::cast<const intervals::Interval&>(domain)
                          ));
    });

    RPY_DBG_ASSERT(self->p_data || !success);

    return new_obj.release().ptr();
}

static void stvs_finalize(PyObject* self)
{
    auto& data = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    if (data) { data.reset(); }
    Py_TYPE(self)->tp_free(self);
}


static PyObject* stvs_query(PyObject* self, PyObject* py_domain)
{
    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        const auto& domain = py::cast<const intervals::Interval&>(py_domain);
        const auto& vs = reinterpret_cast<const RPySimpleTensorValuedStream*>(
            self)->p_data;

        result = python::TensorValuedStream_FromPtr(vs->query(domain));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}


static PyObject* stvs_initial_value(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->initial_value());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}


static PyObject* stvs_terminal_value(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->terminal_value());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_increment_stream(PyObject* self)
{
    py::object result;

    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(Stream(stream->increment_stream())));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_domain(PyObject* self)
{
    py::object result;
    auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->p_data;
    auto success = python::with_caught_exceptions([&]() {
        result = py::cast(stream->domain());
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_repr(PyObject* self) { return PyObject_Repr(self); }

static PyObject* stvs_signature(PyObject* self,
                                PyObject* args,
                                PyObject* kwargs)
{
    python::SigArgs sig_args;

    const auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->
            p_data;
    if (parse_sig_args(args, kwargs, &stream->metadata(), &sig_args)
        < 0) { return nullptr; }

    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        if (!sig_args.interval) { sig_args.interval = stream->domain(); }
        if (!sig_args.resolution) {
            sig_args.resolution = stream->metadata().default_resolution;
        }
        if (!sig_args.ctx) {
            sig_args.ctx = stream->metadata().default_context;
        }

        algebra::FreeTensor sig; {
            py::gil_scoped_release gil;
            sig = stream->signature(*sig_args.interval,
                                    *sig_args.resolution,
                                    *sig_args.ctx);
        }

        result = py::cast(std::move(sig));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}

static PyObject* stvs_log_signature(PyObject* self,
                                    PyObject* args,
                                    PyObject* kwargs)
{
    python::SigArgs sig_args;

    const auto& stream = reinterpret_cast<RPySimpleTensorValuedStream*>(self)->
            p_data;
    if (parse_sig_args(args, kwargs, &stream->metadata(), &sig_args)
        < 0) { return nullptr; }

    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        if (!sig_args.interval) { sig_args.interval = stream->domain(); }
        if (!sig_args.resolution) {
            sig_args.resolution = stream->metadata().default_resolution;
        }
        if (!sig_args.ctx) {
            sig_args.ctx = stream->metadata().default_context;
        }

        algebra::Lie logsig; {
            py::gil_scoped_release gil;
            logsig = stream->log_signature(*sig_args.interval,
                                           *sig_args.resolution,
                                           *sig_args.ctx);
        }

        result = py::cast(std::move(logsig));
    });

    RPY_DBG_ASSERT(result || !success);

    return result.release().ptr();
}


/**
 * Constructs a SimpleTensorValuedStream (STVS) object from the given values and metadata.
 *
 * The function extracts and processes input values from the provided arguments and keyword
 * arguments to build an STVS object. It expects a list of tuples, where each tuple contains
 * a parameter and data in the form of either a group-like free tensor or a Lie element.
 * The function performs data validation, computes the necessary increments, and organizes
 * the resulting metadata for constructing the STVS object.
 *
 * @param cls A Python class object from which the function is invoked.
 * @param args Positional arguments, with args[1] optionally being a sequence of values to parse.
 * @param kwargs Keyword arguments containing optional metadata and the values under the key "values".
 *               Other relevant keyword arguments may be used for providing metadata like resolution, context, etc.
 * @return A PyObject representing the constructed STVS object. If any error occurs (e.g.,
 *         invalid input data or metadata), an exception is raised, and nullptr is returned.
 */
static PyObject* stvs_from_values(PyObject* cls,
                                  PyObject* args,
                                  PyObject* kwargs)
{
    using algebra::Lie;
    auto py_kwargs = py::reinterpret_borrow<py::kwargs>(kwargs);

    py::object value_list;

    auto* tp = reinterpret_cast<PyTypeObject*>(cls);

    if (args != nullptr && PyTuple_Size(args) >= 1) {
        value_list = py::reinterpret_borrow<
            py::object>(PyTuple_GetItem(args, 0));
    } else if (py_kwargs.contains("values")) {
        python::with_caught_exceptions([&]() {
            // This probably never throws, but I'd rather be safe.
            value_list = py_kwargs["values"];
        });
    } else {
        PyErr_SetString(PyExc_ValueError, "expected a list of values");
        return nullptr;
    }

    if (!PySequence_Check(value_list.ptr())) {
        PyErr_SetString(PyExc_TypeError, "expected a sequence of values");
        return nullptr;
    }

    auto size = PySequence_Size(value_list.ptr());
    if (size == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "expected a non-empty sequence of values");
        return nullptr;
    }

    py::object result;

    auto success = python::with_caught_exceptions([&]() {
        std::vector<pair<param_t, FreeTensor> > data;
        data.reserve(size);
        algebra::context_pointer ctx;

        auto path_md = python::kwargs_to_metadata(py_kwargs);
        if (path_md.ctx) { ctx = path_md.ctx; }

        /*
         * To construct the initial data, we need to traverse the sequence of
         * input data and extract the parameter value and the data member. This
         * can be either a group-like free tensor or a Lie. These are decoded
         * into their raw types (converting if necessary) and placed in the
         * data vector. This will be sorted later.
         */
        for (Py_ssize_t i = 0; i < size; ++i) {
            py::handle py_item(PySequence_ITEM(value_list.ptr(), i));

            if (!py::isinstance<py::tuple>(py_item) || py::len(py_item) != 2) {
                throw py::type_error("expected a tuple of length 2");
            }

            py::handle py_param(PySequence_ITEM(py_item.ptr(), 0));
            py::handle py_data(PySequence_ITEM(py_item.ptr(), 1));

            auto param = py::cast<param_t>(py_param);

            if (py::isinstance<FreeTensor>(py_data)) {
                const auto& tensor = py::cast<const FreeTensor&>(py_data);
                if (!ctx) { ctx = tensor.context(); } else {
                    RPY_CHECK_EQ(ctx->width(), tensor.width(), py::value_error);
                    RPY_CHECK_EQ(ctx->ctype(),
                                 tensor.coeff_type(),
                                 py::type_error);
                }
                data.emplace_back(param, py::cast<FreeTensor>(py_data));
            } else if (py::isinstance<Lie>(py_data)) {
                const auto& lie = py::cast<const Lie&>(py_data);
                if (!ctx) { ctx = lie.context(); } else {
                    RPY_CHECK_EQ(ctx->width(), lie.width(), py::value_error);
                    RPY_CHECK_EQ(ctx->ctype(),
                                 lie.coeff_type(),
                                 py::type_error);
                }
                data.emplace_back(param, ctx->lie_to_tensor(lie).exp());
            } else {
                throw py::type_error(
                    "expected a group-like free tensor or Lie data");
            }
        }
        RPY_CHECK_EQ(path_md.width, ctx->width(), py::value_error);

        // There is no reason to expect the values are provided in order.
        std::sort(data.begin(),
                  data.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        // Now it makes sense to get the first element as the initial value.
        FreeTensor initial_value = data.front().second;

        /*
         * The next step is to build a sequence of increments from the value data.
         * This is a list of Lies formed by taking the difference between
         * adjacent elements of the raw data. At the same time, we can compute
         * the smallest difference between adjacent timestamps. We will need
         * this to calculate an appropriate resolution if one isn't given.
         */
        std::vector<pair<param_t, Lie> > increment_data;
        increment_data.reserve(data.size() - 1);
        param_t min_difference = std::numeric_limits<param_t>::infinity();

        // auto previous = ctx->tensor_to_lie(initial_value.log());
        FreeTensor previous = initial_value;
        for (Py_ssize_t i = 1; i < size; ++i) {
            param_t param_diff = data[i].first - data[i - 1].first;
            if (param_diff < min_difference) { min_difference = param_diff; }

            auto increment = previous.antipode().mul(data[i].second);
            increment_data.emplace_back(data[i].first, ctx->tensor_to_lie(increment.log()));
            previous = data[i].second;

            // auto current = ctx->tensor_to_lie(data[i].second.log());

            // increment_data.emplace_back(data[i].first, current.sub(previous));
            // previous = std::move(current);
        }

        /*
         * If resolution was not provided, we should compute the one that
         * separates points.
         */
        if (!path_md.resolution) {
            path_md.resolution = param_to_resolution(min_difference) + 1;
        }

        /*
         * If support is not given, set it to be the smallest interval that
         * contains dyadic intervals of the given resolution containing each of
         * the data points. This is done so the end increment is also included
         * when because the clopen interval
         * [data.front().first, data.back().first) would never show the
         * increment that happens at data.back().first.
         */
        if (!path_md.support) {
            path_md.support = intervals::RealInterval(
                data.front().first,
                data.back().first + ldexp(1., -*path_md.resolution));
        }

        /*
         * I don't like this but we have to do it because the constructor for
         * Lie increment streams still needs the metadata like this.
         */
        auto schema = std::make_shared<StreamSchema>(path_md.width);
        StreamMetadata metadata{
                path_md.width,
                std::move(path_md.support).value(),
                path_md.ctx,
                path_md.scalar_type,
                (path_md.vector_type
                    ? *path_md.vector_type
                    : algebra::VectorType::Dense),
                *path_md.resolution
        };

        auto increment_stream = std::make_shared<LieIncrementStream>(
            std::move(increment_data),
            std::move(metadata),
            std::move(schema));

        // Finally build the result.
        result = py::reinterpret_steal<py::object>(tp->tp_alloc(tp, 0));
        if (!result) { throw py::error_already_set(); }

        auto* self = reinterpret_cast<RPySimpleTensorValuedStream*>(result.
            ptr());

        construct_inplace(&self->p_data,
                          make_simple_tensor_valued_stream(
                              std::move(increment_stream),
                              std::move(initial_value),
                              *path_md.support));
    });

    RPY_DBG_ASSERT(result || !success);
    return result.release().ptr();
}

}


// Define the method table for RPySimpleTensorValuedStream
static PyMethodDef stvs_methods[] = {
        {"query", reinterpret_cast<PyCFunction>(stvs_query), METH_O,
         "Query the stream with a given domain"},
        {"initial_value", reinterpret_cast<PyCFunction>(stvs_initial_value),
         METH_NOARGS, "Get the initial value of the stream"},
        {"terminal_value", reinterpret_cast<PyCFunction>(stvs_terminal_value),
         METH_NOARGS, "Get the terminal value of the stream"},
        {"increment_stream",
         reinterpret_cast<PyCFunction>(stvs_increment_stream), METH_NOARGS,
         "Get the increment stream"},
        {"domain", reinterpret_cast<PyCFunction>(stvs_domain), METH_NOARGS,
         "Get the domain of the stream"},
        {"signature", reinterpret_cast<PyCFunction>(stvs_signature),
         METH_VARARGS | METH_KEYWORDS, "Get the signature of the stream"},
        {"log_signature", reinterpret_cast<PyCFunction>(stvs_log_signature),
         METH_VARARGS | METH_KEYWORDS, "Get the log signature of the stream"},
        {"from_values", reinterpret_cast<PyCFunction>(stvs_from_values),
         METH_VARARGS | METH_KEYWORDS | METH_CLASS,
         "Construct a new tensor-valued stream from a sequence of group-like free tensors or Lie elements"},
        {nullptr, nullptr, 0, nullptr}// Sentinel
};

// Define the PyTypeObject for RPySimpleTensorValuedStream
PyTypeObject python::TensorValuedStream_Type = {
        PyVarObject_HEAD_INIT(nullptr, 0)
        "roughpy.TensorValuedStream",// tp_name
        sizeof(RPySimpleTensorValuedStream),// tp_basicsize
        0,// tp_itemsize
        nullptr,// tp_dealloc
        0,// tp_vectorcall_offset
        0,// tp_getattr
        0,// tp_setattr
        0,// tp_as_async
        stvs_repr,// tp_repr
        0,// tp_as_number
        0,// tp_as_sequence
        0,// tp_as_mapping
        0,// tp_hash
        0,// tp_call
        0,// tp_str
        0,// tp_getattro
        0,// tp_setattro
        0,// tp_as_buffer
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_FINALIZE,
        // tp_flags
        "A Simple Tensor Valued stream",// tp_doc
        0,// tp_traverse
        0,// tp_clear
        0,// tp_richcompare
        0,// tp_weaklistoffset
        0,// tp_iter
        0,// tp_iternext
        stvs_methods,// tp_methods
        0,// tp_members
        0,// tp_getset
        0,// tp_base
        0,// tp_dict
        0,// tp_descr_get
        0,// tp_descr_set
        0,// tp_dictoffset
        0,// tp_init
        0,// tp_alloc
        stvs_new,// tp_new
        0,// tp_free
        0,// tp_is_gc
        0,// tp_bases
        0,// tp_mro
        0,// tp_cache
        0,// tp_subclasses
        0,// tp_weaklist
        0,// tp_del
        0,// tp_version_tag
        stvs_finalize// tp_finalize
};

void python::init_tensor_valued_stream(py::module_& m)
{
    if (PyType_Ready(&TensorValuedStream_Type) < 0) {
        throw py::error_already_set();
    }

    m.add_object("TensorValuedStream",
                 reinterpret_cast<PyObject*>(&
                     TensorValuedStream_Type));

}