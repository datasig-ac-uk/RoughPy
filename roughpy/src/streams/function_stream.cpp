#include "function_stream.h"
using namespace rpy;


static const char* FUNC_STREAM_DOC = R"rpydoc(A stream generated dynamically by calling a function.
)rpydoc";



python::FunctionStream::FunctionStream(py::function fn, streams::StreamMetadata md)
    : DynamicallyConstructedStream(std::move(md)), m_fn(std::move(fn))
{
}
algebra::Lie python::FunctionStream::log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const {
    return ctx.zero_lie(metadata().cached_vector_type);
}

void python::init_function_stream(py::module_ &m) {

    py::class_<FunctionStream> klass(m, "FunctionStream", FUNC_STREAM_DOC);

}
