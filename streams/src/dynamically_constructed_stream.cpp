//
// Created by user on 18/03/23.
//

#include "dynamically_constructed_stream.h"

using namespace rpy;
using namespace rpy::streams;

algebra::Lie DynamicallyConstructedStream::log_signature(const intervals::Interval &interval, const algebra::Context &ctx) const {
    //TODO: Implement this properly
    return eval(interval);
}
