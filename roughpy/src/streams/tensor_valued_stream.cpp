//
// Created by sammorley on 10/12/24.
//

#include "tensor_valued_stream.h"


#include "roughpy/algebra/free_tensor.h"
#include "roughpy/streams/value_stream.h"

#include "stream.h"

using namespace pybind11::literals;
using namespace rpy;
using namespace rpy::streams;

using algebra::FreeTensor;


namespace {


std::shared_ptr<const ValueStream<FreeTensor>>
simple_tensor_valued_stream_pyinit(py::handle py_increment_stream,
                                   py::handle py_initial_value,
                                   py::handle py_domain)
{

    if (isinstance(py_increment_stream,
                   reinterpret_cast<PyObject*>(&python::RPyStream_Type))) {
        RPY_THROW(py::type_error, "increment stream must be a stream object");
    }

    const auto& increment_stream_wrapper = reinterpret_cast<const
        python::RPyStream*>(py_increment_stream.ptr())->m_data;

    const auto& initial_value = py::cast<const FreeTensor&>(py_initial_value);
    const auto& domain = py::cast<const intervals::Interval&>(py_domain);

    return make_simple_tensor_valued_stream(increment_stream_wrapper.impl(),
                                            initial_value,
                                            domain);
}


}


void python::init_tensor_valued_stream(py::module_& m)
{
    py::class_<ValueStream<FreeTensor>, StreamInterface, std::shared_ptr<const
        ValueStream<FreeTensor>> > klass(m, "SimpleTensorValuedStream");

    klass.def(py::init(&simple_tensor_valued_stream_pyinit),
              "increment_stream"_a,
              "initial_value"_a,
              "domain"_a);

    klass.def("increment_stream",
              [](const std::shared_ptr<const ValueStream<FreeTensor>>& self) {
                  return RPyStream_FromStream(Stream(self->increment_stream()));
              });

    klass.def("initial_value",
              [](const std::shared_ptr<const ValueStream<FreeTensor>>& self) {
                  return self->initial_value();
              });

    klass.def("domain",
              [](const std::shared_ptr<const ValueStream<FreeTensor>>& self) {
                  return self->domain();
              });

    klass.def("terminal_value",
              [](const std::shared_ptr<const ValueStream<FreeTensor>>& self) {
                  return self->terminal_value();
              });

    klass.def("query", &ValueStream<FreeTensor>::query, "interval"_a);
}