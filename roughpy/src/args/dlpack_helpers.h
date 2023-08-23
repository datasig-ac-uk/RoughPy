//
// Created by sam on 11/08/23.
//

#ifndef ROUGHPY_DLPACK_HELPERS_H
#define ROUGHPY_DLPACK_HELPERS_H

#include "roughpy_module.h"
#include "dlpack.h"

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

constexpr platform::DeviceType
convert_from_dl_device_type(DLDeviceType type) noexcept {
    return static_cast<platform::DeviceType>(type);
}

constexpr DLDeviceType
convert_to_dl_device_type(platform::DeviceType type) noexcept
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

constexpr platform::DeviceInfo
convert_from_dl_device_info(const DLDevice& device) noexcept
{
    return {
        convert_from_dl_device_type(device.device_type),
        device.device_id
    };
}

constexpr DLDevice
convert_to_dl_device_info(const platform::DeviceInfo& device) noexcept {
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
