#ifndef ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
#define ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_

#include "stream_base.h"


namespace rpy {
namespace streams {

class DynamicallyConstructedStream : public StreamInterface {
public:

    using StreamInterface::StreamInterface;

    algebra::Lie log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const override;

protected:
    virtual algebra::Lie eval(const intervals::Interval& interval) const = 0;
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_DYNAMICALLY_CONSTRUCTED_STREAM_H_
