//
// Created by sam on 08/08/23.
//

#include <roughpy/scalars/key_scalar_stream.h>
#include <roughpy/scalars/types.h>

using namespace rpy;
using namespace rpy::scalars;

KeyScalarStream::KeyScalarStream() = default;
KeyScalarStream::KeyScalarStream(const rpy::scalars::KeyScalarStream&)
        = default;
KeyScalarStream::KeyScalarStream(rpy::scalars::KeyScalarStream&&) noexcept
        = default;

KeyScalarStream& KeyScalarStream::operator=(const KeyScalarStream&) = default;
KeyScalarStream& KeyScalarStream::operator=(KeyScalarStream&&) noexcept
        = default;

KeyScalarStream::KeyScalarStream(const ScalarType* type) : ScalarStream(type)
{}

void KeyScalarStream::reserve_size(dimn_t num_rows)
{
    ScalarStream::reserve_size(num_rows);
    m_key_stream.reserve(num_rows);
}
void KeyScalarStream::push_back(
        const ScalarPointer& scalar_ptr, const key_type* key_ptr
)
{
    if (key_ptr != nullptr) {
        if (m_key_stream.empty()) {
            m_key_stream.resize(m_stream.size(), nullptr);
        }
        m_key_stream.push_back(key_ptr);
    }
    ScalarStream::push_back(scalar_ptr);
}
void KeyScalarStream::push_back(
        const ScalarArray& scalar_data, const key_type* key_ptr
)
{
    if (key_ptr != nullptr) {
        if (m_key_stream.empty()) {
            m_key_stream.resize(m_stream.size(), nullptr);
        }
        m_key_stream.push_back(key_ptr);
    }
    ScalarStream::push_back(scalar_data);
}

KeyScalarArray KeyScalarStream::operator[](dimn_t row) const noexcept
{
    return {
        ScalarStream::operator[](row),
        m_key_stream.empty() ? nullptr : m_key_stream[row]
    };
}
