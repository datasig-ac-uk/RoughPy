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

#include "roughpy/core/check.h"             // for throw_exception, RPY_CHECK
#include "roughpy/core/macros.h"            // for RPY_UNREACHABLE_RETURN
#include "roughpy/core/types.h"             // for uint8_t, string

#include <roughpy/core/helpers.h>
#include <roughpy/scalars/scalars_fwd.h>
#include <roughpy/scalars/scalar_type.h>

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

constexpr devices::DeviceType
convert_from_dl_device_type(DLDeviceType type) noexcept {
    return static_cast<devices::DeviceType>(type);
}

constexpr DLDeviceType
convert_to_dl_device_type(devices::DeviceType type) noexcept
{
    return static_cast<DLDeviceType>(type);
}

constexpr scalars::BasicScalarInfo
convert_from_dl_datatype(const DLDataType& dtype) noexcept {
    return {
        convert_from_dl_typecode(dtype.code),
        static_cast<uint8_t>(dtype.bits / CHAR_BIT),
        round_up_divide(dtype.bits, CHAR_BIT),
        static_cast<uint8_t>(dtype.lanes & 0xFF)
    };
}

inline DLDataType
convert_to_dl_datatype(const scalars::BasicScalarInfo& info) noexcept
{
    return {
        convert_to_dl_typecode(info.code),
        static_cast<uint8_t>(info.bytes * CHAR_BIT),
        info.lanes
    };
}

constexpr devices::DeviceInfo
convert_from_dl_device_info(const DLDevice& device) noexcept
{
    return {
        convert_from_dl_device_type(device.device_type),
        device.device_id
    };
}

constexpr DLDevice
convert_to_dl_device_info(const devices::DeviceInfo& device) noexcept {
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
    auto dtype_host = scalars::scalar_type_of(convert_from_dl_datatype(dtype));

    RPY_CHECK(dtype_host);

    switch (device.device_type) {
        case kDLCPU:
            return *dtype_host;
        case kDLCUDA:
        case kDLCUDAHost:
        case kDLOpenCL:
        case kDLVulkan:
        case kDLMetal:
        case kDLVPI:
        case kDLROCM:
        case kDLROCMHost:
        case kDLExtDev:
        case kDLCUDAManaged:
        case kDLOneAPI:
        case kDLWebGPU:
        case kDLHexagon:
            RPY_THROW(std::invalid_argument, "devices are not currently supported");
    }

    RPY_UNREACHABLE_RETURN(nullptr);
}

RPY_NO_EXPORT inline py::capsule py_to_dlpack(py::handle arg)
{
    return py::reinterpret_borrow<py::capsule>(arg.attr("__dlpack__")(
    ));
}

RPY_NO_EXPORT inline DLManagedTensor*
unpack_dl_capsule(const py::capsule& cap) noexcept
{
    return cap.get_pointer<DLManagedTensor>();
}

}
}

#endif// ROUGHPY_DLPACK_HELPERS_H
