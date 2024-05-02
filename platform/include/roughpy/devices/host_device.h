// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_HOST_DEVICE_H_
#define ROUGHPY_DEVICE_HOST_DEVICE_H_

#include "device_handle.h"

namespace rpy {
namespace devices {

/**
 * @class HostDeviceHandle
 * @brief Represents a handle to a host device.
 *
 * This class is a subclass of DeviceHandle and represents a handle to a host
 * device. It provides the functionality to compute delegate to the host device.
 */
class ROUGHPY_DEVICES_EXPORT HostDeviceHandle : public DeviceHandle
{

public:
    /**
     * @brief Computes the delegate for the host device.
     *
     * This pure virtual method returns a Device object representing the
     * delegate for the host device.
     *
     * @return Device A Device object representing the delegate for the host
     * device.
     *
     * @see HostDeviceHandle
     */
    RPY_NO_DISCARD virtual Device compute_delegate() const = 0;
};
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_HOST_DEVICE_H_