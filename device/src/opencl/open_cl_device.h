//
// Created by sam on 25/08/23.
//

#ifndef ROUGHPY_OPEN_CL_DEVICE_H
#define ROUGHPY_OPEN_CL_DEVICE_H

#include <roughpy/device/core.h>
#include <roughpy/device/device_handle.h>
#include <roughpy/device/kernel.h>

#include "open_cl_runtime_library.h"


namespace rpy {
namespace device {

class OpenCLDevice : public DeviceHandle
{
    OpenCLRuntimeLibrary* p_runtime;




};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_OPEN_CL_DEVICE_H
