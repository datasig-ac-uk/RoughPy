//
// Created by sam on 3/28/24.
//

#include "key_scalar_stream.h"

#include <roughpy/core/ranges.h>

#include <roughpy/devices/algorithms.h>

rpy::dimn_t rpy::algebra::KeyScalarStream::col_count(dimn_t i) const noexcept
{
    RPY_DBG_ASSERT(i < m_parts.size());
    return m_parts[i].size();
}
rpy::dimn_t rpy::algebra::KeyScalarStream::max_col_count() const noexcept
{
    return empty()
            ? 0
            : ranges::max(m_parts | views::transform([](const auto& ksa) {
                              return ksa.size();
                          }));
}
rpy::algebra::KeyScalarArray rpy::algebra::KeyScalarStream::operator[](dimn_t i
) const noexcept
{
    RPY_DBG_ASSERT(i < m_parts.size());
    return KeyScalarArray(m_parts[i]);
}
rpy::algebra::KeyScalarStream
rpy::algebra::KeyScalarStream::operator[](SliceIndex indices) const noexcept
{
    return KeyScalarStream();
}
void rpy::algebra::KeyScalarStream::set_type(scalars::TypePtr type) noexcept
{
    if (!m_parts.empty()) {}

    p_type = std::move(type);
}
void rpy::algebra::KeyScalarStream::reserve(dimn_t num_rows)
{
    m_parts.reserve(num_rows);
}
void rpy::algebra::KeyScalarStream::push_back(const scalars::ScalarArray& data)
{
    push_back(scalars::ScalarArray(data));
}
void rpy::algebra::KeyScalarStream::push_back(scalars::ScalarArray&& data)
{
    push_back(KeyScalarArray(std::move(data)));
}
void rpy::algebra::KeyScalarStream::push_back(const KeyScalarArray& data)
{
    if (p_type == nullptr) {
        RPY_DBG_ASSERT(m_parts.empty());
        p_type = data.type();
    }

    auto& back = m_parts.emplace_back(p_type, data.size());
    devices::algorithms::copy(back, data);

    if (data.has_keys()) { back.keys() = data.keys(); }
}
void rpy::algebra::KeyScalarStream::push_back(KeyScalarArray&& data)
{
    if (p_type == nullptr) {
        RPY_DBG_ASSERT(m_parts.empty());
        p_type = data.type();
    }

    if (data.type() == p_type) {
        m_parts.emplace_back(std::move(data));
    } else {
        auto& back = m_parts.emplace_back(p_type, data.size());
        devices::algorithms::copy(back, data);
        if (data.has_keys()) { back.keys() = std::move(data.keys()); }
    }
}
