//
// Created by sam on 06/02/24.
//

#ifndef ROUGHPY_STREAMS_FWD_H
#define ROUGHPY_STREAMS_FWD_H

#include <roughpy/core/types.h>

#include "roughpy_streams_export.h"

namespace rpy { namespace streams {

enum struct ChannelType : uint8_t
{
    Increment = 0,
    Value = 1,
    Categorical = 2,
    Lie = 3,
};

}}

#endif// ROUGHPY_STREAMS_FWD_H
