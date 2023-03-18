//
// Created by sam on 18/03/23.
//

#include "BaseStream.h"


using namespace rpy;
using namespace rpy::python;

static const char * STREAM_INTERFACE_DOC = R"rpydoc(The stream interface is the means by which one converts
an example of streaming data into a rough path.
)rpydoc";


void python::init_base_stream(py::module_ &m) {

    py::class_<streams::StreamInterface, PyBaseStream> klass(m, "StreamInterface", STREAM_INTERFACE_DOC);

    // TODO: Finish this off.


}

algebra::Lie PyBaseStream::log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    PYBIND11_OVERRIDE_PURE(algebra::Lie, streams::StreamInterface, log_signature, interval, ctx);
}
bool PyBaseStream::empty(const intervals::Interval &interval) const noexcept {
    PYBIND11_OVERRIDE(bool, streams::StreamInterface, empty, interval);
}
algebra::Lie PyBaseStream::log_signature(const intervals::DyadicInterval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const {
    PYBIND11_OVERRIDE(algebra::Lie, streams::StreamInterface, log_signature, interval, resolution, ctx);
}
algebra::Lie PyBaseStream::log_signature(const intervals::Interval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const {
    PYBIND11_OVERRIDE(algebra::Lie, streams::StreamInterface, log_signature, interval, resolution, ctx);
}
algebra::FreeTensor PyBaseStream::signature(const intervals::Interval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const {
    PYBIND11_OVERRIDE(algebra::FreeTensor, streams::StreamInterface, signature, interval, resolution, ctx);
}
