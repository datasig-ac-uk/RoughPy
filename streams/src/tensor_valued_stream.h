//
// Created by sam on 05/12/24.
//

#ifndef ROUGHPY_STREAMS_TENSOR_VALUED_STREAM_H
#define ROUGHPY_STREAMS_TENSOR_VALUED_STREAM_H


#include <functional>
#include <memory>


#include "roughpy/generics/type.h"

#include "value_stream.h"


namespace rpy {
namespace streams {




class TensorValuedStream : public ValueStream<algebra::FreeTensor> {
protected:
    using ValueStream::traits;

private:
    intervals::RealInterval m_domain;
    std::shared_ptr<const StreamInterface> p_increment_stream;
    StreamValue m_initial_value;

public:
    TensorValuedStream(
        intervals::RealInterval domain,
        std::shared_ptr<const StreamInterface> increment_stream,
        StreamValue initial_value
    );
    RPY_NO_DISCARD bool
    empty(const intervals::Interval& interval) const noexcept override;

protected:
    RPY_NO_DISCARD algebra::Lie log_signature_impl(
        const intervals::Interval& interval,
        const algebra::Context& ctx) const override;

public:
    const intervals::RealInterval& domain() const noexcept override;

    RPY_NO_DISCARD std::shared_ptr<const ValueStream> query(
        const intervals::Interval& interval) const override;

    RPY_NO_DISCARD std::shared_ptr<const StreamInterface>
    increment_stream() const noexcept override;

    RPY_NO_DISCARD StreamValue value_at(param_t param) const override;

    RPY_NO_DISCARD StreamValue initial_value() const override;

    RPY_NO_DISCARD StreamValue terminal_value() const override;


};

} // streams
} // rpy

#endif //ROUGHPY_STREAMS_TENSOR_VALUED_STREAM_H
