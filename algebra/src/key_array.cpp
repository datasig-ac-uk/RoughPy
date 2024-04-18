//
// Created by sam on 3/18/24.
//

#include "key_array.h"
#include <roughpy/scalars/devices/host_device.h>

using namespace rpy;
using namespace rpy::algebra;

KeyArray::KeyArray() = default;
KeyArray::KeyArray(const KeyArray&) = default;
KeyArray::KeyArray(rpy::algebra::KeyArray&&) noexcept = default;

KeyArray::KeyArray(dimn_t size)
    : m_buffer(devices::get_host_device()->alloc(basis_key_type_info, size))
{}

KeyArray::KeyArray(devices::Device device, dimn_t size)
    : m_buffer(device->alloc(basis_key_type_info, size))
{}

KeyArray::KeyArray(Slice<BasisKey> keys)
    : m_buffer(keys.data(), keys.size(), basis_key_type_info)
{}

KeyArray::~KeyArray()
{
    if (!m_buffer.is_null() && m_buffer.is_host() && m_buffer.is_owner()) {
        auto slice = m_buffer.as_mut_slice<BasisKey>();
        std::destroy(slice.begin(), slice.end());
    }
}

KeyArray& KeyArray::operator=(const KeyArray&) = default;
KeyArray& KeyArray::operator=(KeyArray&&) noexcept = default;

BasisKey KeyArray::operator[](dimn_t index) const
{
    RPY_CHECK(index < m_buffer.size());
    if (!m_buffer.is_host()) {
        auto tmp_buffer = m_buffer.map(1, index);
        // If the key is on a device, it must be a trivial type.
        return tmp_buffer.as_slice<BasisKey>()[0];
    }

    return as_slice()[index];
}

BasisKey& KeyArray::operator[](dimn_t index)
{
    RPY_CHECK(index < m_buffer.size());
    if (!m_buffer.is_host()) {
        RPY_THROW(
                std::runtime_error,
                "cannot access keys from non-host buffer"
        );
    }

    return m_buffer.as_mut_slice<BasisKey>()[index];
}
KeyArray KeyArray::to_device(devices::Device device) const
{
    devices::Buffer new_buffer
            = device->alloc(basis_key_type_info, this->size());
    m_buffer.to_device(new_buffer, device);
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
    return KeyArray(m_buffer.slice(offset, size));
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
    return KeyArray(m_buffer.slice(offset, size));
}
