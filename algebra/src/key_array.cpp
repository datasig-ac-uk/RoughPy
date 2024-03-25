//
// Created by sam on 3/18/24.
//

#include "key_array.h"

using namespace rpy;
using namespace rpy::algebra;

KeyArray::KeyArray() = default;
KeyArray::KeyArray(const KeyArray&) = default;
KeyArray::KeyArray(rpy::algebra::KeyArray&&) noexcept = default;

KeyArray::KeyArray(Slice<BasisKey> keys)
    : m_buffer(keys.data(), keys.size(), basis_key_type_info)
{}

KeyArray::~KeyArray()
{
    if (m_buffer.is_host() && m_buffer.is_owner()) {
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
