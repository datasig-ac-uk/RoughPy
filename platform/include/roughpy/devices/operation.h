//
// Created by sam on 7/11/24.
//

#ifndef ROUGHPY_DEVICES_OPERATION_H
#define ROUGHPY_DEVICES_OPERATION_H

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "buffer.h"
#include "core.h"
#include "kernel.h"
#include "kernel_operators.h"
#include "kernel_parameters.h"
#include "queue.h"

namespace rpy {
namespace devices {

class ROUGHPY_DEVICES_EXPORT Operation
{



    virtual ~Operation() = default;
};



}// namespace devices
}// namespace rpy
#endif// ROUGHPY_DEVICES_OPERATION_H
