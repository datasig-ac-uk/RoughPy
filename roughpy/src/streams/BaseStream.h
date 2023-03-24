//
// Created by sam on 18/03/23.
//

#ifndef ROUGHPY_BASESTREAM_H
#define ROUGHPY_BASESTREAM_H

#include "roughpy_module.h"
#include <roughpy/streams/stream_base.h>

namespace rpy {
namespace python {


class PyBaseStream : public streams::StreamInterface {
public:
    algebra::Lie log_signature_impl(const intervals::Interval &interval, const algebra::Context &ctx) const override;
    bool empty(const intervals::Interval &interval) const noexcept override;
    algebra::Lie log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const override;
    algebra::FreeTensor signature(const intervals::Interval &interval, const algebra::Context &ctx) const override;
    algebra::Lie log_signature(const intervals::DyadicInterval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const override;
    algebra::Lie log_signature(const intervals::Interval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const override;
    algebra::FreeTensor signature(const intervals::Interval &interval, streams::resolution_t resolution, const algebra::Context &ctx) const override;
};


void init_base_stream(py::module_& m);


}// namespace python
}// namespace rpy

#endif//ROUGHPY_BASESTREAM_H
