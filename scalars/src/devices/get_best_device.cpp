//
// Created by sam on 4/17/24.
//

#include "devices/core.h"

#include "devices/device_handle.h"
#include "devices/host_device.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

Device devices::get_best_device(Slice<Device> devices)
{

    return ranges::fold_left(
            devices,
            nullptr,
            [](const Device& left, const Device& right) {
                if (!left) { return right; }
                if (left->is_host()) { return left; }
                if (!right) { return left; }

                if (left == right) { return left; }
                return static_cast<Device>(get_host_device());
            }
    );
}
