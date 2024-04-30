//
// Created by sam on 10/04/24.
//

#ifndef KERNEL_REGISTRY_H
#define KERNEL_REGISTRY_H

#include "devices/core.h"
#include "devices/device_handle.h"
#include "devices/host_device.h"
#include "host_kernel.h"

namespace rpy {
namespace devices {

struct RegistrationHelper {
    template <typename... Args>
    explicit RegistrationHelper(Args&& args)
    {
        get_host_device()->register_kernel(
                Kernel(new CPUKernel(std::forward<Args>(args)...))
        );
    }
};

}// namespace devices
}// namespace rpy

#endif// KERNEL_REGISTRY_H
