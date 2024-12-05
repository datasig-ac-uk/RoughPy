//
// Created by sammorley on 03/12/24.
//

#include "value_stream.h"


using namespace rpy;
using namespace rpy::streams;


ValueStream::StreamValue ValueStream::initial_value() const
{
    return value_at(domain().inf());
}

ValueStream::StreamValue ValueStream::terminal_value() const
{
    return value_at(domain().sup());
}
