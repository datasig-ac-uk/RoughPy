//
// Created by sam on 18/08/23.
//

#ifndef ROUGHPY_CPUDEVICE_H
#define ROUGHPY_CPUDEVICE_H

#include <roughpy/platform/device.h>

namespace rpy {
namespace platform {

class CPUDevice : public DeviceHandle
{

public:

    CPUDevice() : DeviceHandle(DeviceType::CPU, 0) {}

    RPY_NO_DISCARD
    optional<fs::path> runtime_library() const noexcept override;

};

}// namespace platform
}// namespace rpy

#endif// ROUGHPY_CPUDEVICE_H
