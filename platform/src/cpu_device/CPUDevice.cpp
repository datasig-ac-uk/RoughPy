//
// Created by sam on 18/08/23.
//

#include "CPUDevice.h"

std::optional<std::filesystem::path>
rpy::platform::CPUDevice::runtime_library() const noexcept
{
    return {};
}
