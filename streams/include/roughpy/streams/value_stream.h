//
// Created by sammorley on 03/12/24.
//

#ifndef ROUGHPY_STREAMS_VALUE_STREAM_H
#define ROUGHPY_STREAMS_VALUE_STREAM_H

#include <memory>

#include <roughpy/core/macros.h>

#include <roughpy/generics/values.h>

#include "roughpy/algebra/free_tensor.h"

#include "arrival_stream.h"
#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {

template <typename T>
struct StreamValueTraits {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
};

template <typename ValueType>
class ValueStream : public StreamInterface
{
protected:
    using traits = StreamValueTraits<ValueType>;
public:
    using StreamValue = typename traits::value_type;

    using StreamInterface::StreamInterface;

    virtual const intervals::RealInterval& domain() const noexcept = 0;

    RPY_NO_DISCARD
    virtual std::shared_ptr<const ValueStream> query(
        const intervals::Interval& interval) const = 0;

    RPY_NO_DISCARD
    virtual std::shared_ptr<const StreamInterface>
    increment_stream() const noexcept = 0;

    RPY_NO_DISCARD
    virtual StreamValue value_at(param_t param) const = 0;

    RPY_NO_DISCARD
    virtual StreamValue initial_value() const = 0;

    RPY_NO_DISCARD
    virtual StreamValue terminal_value() const = 0;
};



template class ROUGHPY_STREAMS_EXPORT ValueStream<algebra::FreeTensor>;


ROUGHPY_STREAMS_EXPORT
std::shared_ptr<const ValueStream<algebra::FreeTensor>>
make_simple_tensor_valued_stream(
    std::shared_ptr<const StreamInterface> increment_stream,
    algebra::FreeTensor initial_value,
    const intervals::Interval& domain
    );

}// streams
}// rpy

#endif //ROUGHPY_STREAMS_VALUE_STREAM_H