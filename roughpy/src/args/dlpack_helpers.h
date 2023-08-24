// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by sam on 11/08/23.
//

#ifndef ROUGHPY_DLPACK_HELPERS_H
#define ROUGHPY_DLPACK_HELPERS_H

#include "roughpy_module.h"
#include "dlpack.h"

#include <roughpy/scalars/scalars_fwd.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/device/core.h>

namespace rpy {
namespace python {

constexpr scalars::ScalarTypeCode
convert_from_dl_typecode(uint8_t code) noexcept {
    return static_cast<scalars::ScalarTypeCode>(code);
}
constexpr uint8_t
convert_to_dl_typecode(scalars::ScalarTypeCode code) noexcept {
    return static_cast<uint8_t>(code);
}

constexpr device::DeviceType
convert_from_dl_device_type(DLDeviceType type) noexcept {
    return static_cast<device::DeviceType>(type);
}

constexpr DLDeviceType
convert_to_dl_device_type(device::DeviceType type) noexcept
{
    return static_cast<DLDeviceType>(type);
}

constexpr scalars::BasicScalarInfo
convert_from_dl_datatype(const DLDataType& dtype) noexcept {
    return {
        convert_from_dl_typecode(dtype.code),
        dtype.bits,
        dtype.lanes
    };
}

inline DLDataType
convert_to_dl_datatype(const scalars::BasicScalarInfo& info) noexcept
{
    return {
        convert_to_dl_typecode(info.code),
        info.bits,
        info.lanes
    };
}

constexpr device::DeviceInfo
convert_from_dl_device_info(const DLDevice& device) noexcept
{
    return {
        convert_from_dl_device_type(device.device_type),
        device.device_id
    };
}

constexpr DLDevice
convert_to_dl_device_info(const device::DeviceInfo& device) noexcept {
    return {
        convert_to_dl_device_type(device.device_type),
        device.device_id
    };
}


const string&
type_id_for_dl_info(const DLDataType& dtype, const DLDevice& device);

const scalars::ScalarType*
scalar_type_for_dl_info(const DLDataType& dtype, const DLDevice& device);

inline const scalars::ScalarType* scalar_type_of_dl_info(const DLDataType& dtype, const DLDevice& device)
{
    return scalars::ScalarType::from_type_details(
            convert_from_dl_datatype(dtype),
            convert_from_dl_device_info(device));
}

}
}

#endif// ROUGHPY_DLPACK_HELPERS_H
