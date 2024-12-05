//
// Created by sammorley on 03/12/24.
//

#ifndef ROUGHPY_STREAMS_VALUE_STREAM_H
#define ROUGHPY_STREAMS_VALUE_STREAM_H

#include <memory>

#include <roughpy/core/macros.h>

#include <roughpy/generics/values.h>

#include "arrival_stream.h"

namespace rpy {
namespace streams {

class ValueStream : public StreamInterface
{
public:
    using StreamValue = generics::Value;

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
    virtual StreamValue initial_value() const;

    RPY_NO_DISCARD
    virtual StreamValue terminal_value() const;

};

}// streams
}// rpy

#endif //ROUGHPY_STREAMS_VALUE_STREAM_H