//
// Created by sam on 3/18/24.
//

#include "key_array.h"
#include <roughpy/devices/host_device.h>

using namespace rpy;
using namespace rpy::algebra;

BasisKey KeyArray::operator[](dimn_t index) const
{
    RPY_CHECK(index < size());
    if (!is_host()) {
        auto tmp_buffer = map(1, index);
        // If the key is on a device, it must be a trivial type.
        return tmp_buffer.as_slice<BasisKey>()[0];
    }

    // return as_slice()[index];
    return {};
}

BasisKey& KeyArray::operator[](dimn_t index)
{
    RPY_CHECK(index < size());
    if (!is_host()) {
        RPY_THROW(
                std::runtime_error,
                "cannot access keys from non-host buffer"
        );
    }

    return as_mut_slice<BasisKey>()[index];
}
KeyArray KeyArray::to_device(devices::Device device) const
{
    Buffer new_buffer = device->alloc(*type(), this->size());
    Buffer::to_device(new_buffer, device);
    return KeyArray(std::move(new_buffer));
}

KeyArray KeyArray::operator[](SliceIndex index)
{
    RPY_DBG_ASSERT(index.begin < index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );

    const auto offset = index.begin * sizeof(BasisKey);
    const auto size = (index.end - index.begin) * sizeof(BasisKey);
    return KeyArray(slice(offset, size));
}
KeyArray KeyArray::operator[](SliceIndex index) const
{
    RPY_DBG_ASSERT(index.begin < index.end);
    const auto buffer_size = size();
    RPY_CHECK(
            index.end <= buffer_size,
            "index end " + std::to_string(index.end)
                    + " is out of bounds for array of size "
                    + std::to_string(buffer_size)
    );

    const auto offset = index.begin * sizeof(BasisKey);
    const auto size = (index.end - index.begin) * sizeof(BasisKey);
    return KeyArray(slice(offset, size));
}
