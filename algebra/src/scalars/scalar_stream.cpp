//
// Created by sam on 25/06/24.
//

#include "scalar_stream.h"

using namespace rpy;
using namespace rpy::scalars;

ScalarStream::ScalarStream() : m_stream(), p_type(nullptr) {}
ScalarStream::ScalarStream(const ScalarStream& other)
    : m_stream(other.m_stream),
      p_type(other.p_type)
{}
ScalarStream::ScalarStream(ScalarStream&& other) noexcept
    : m_stream(std::move(other.m_stream)),
      p_type(std::move(other.p_type))
{}
ScalarStream::ScalarStream(TypePtr type) : m_stream(), p_type(std::move(type))
{}
ScalarStream::ScalarStream(ScalarArray base, containers::Vec<dimn_t> shape)
{
    auto tp = base.type();
    p_type = tp;

    if (shape.empty()) {
        RPY_THROW(std::runtime_error, "strides cannot be empty");
    }

    dimn_t rows = shape[0];
    dimn_t cols = (shape.size() > 1) ? shape[1] : 1;

    m_stream.reserve(rows);

    for (dimn_t i = 0; i < rows; ++i) {
        m_stream.push_back(base[{i * cols, (i + 1) * cols}]);
    }
}
ScalarStream& ScalarStream::operator=(const ScalarStream& other)
{
    if (&other != this) {
        this->~ScalarStream();
        m_stream = other.m_stream;
        p_type = other.p_type;
    }
    return *this;
}
ScalarStream& ScalarStream::operator=(ScalarStream&& other) noexcept
{
    if (&other != this) {
        this->~ScalarStream();
        m_stream = std::move(other.m_stream);
        p_type = other.p_type;
    }
    return *this;
}
dimn_t ScalarStream::col_count(dimn_t i) const noexcept
{
    RPY_CHECK(i < m_stream.size());
    return m_stream[i].size();
}
dimn_t ScalarStream::max_row_size() const noexcept
{
    if (m_stream.empty()) { return 0; }

    containers::Vec<dimn_t> tmp;
    tmp.reserve(m_stream.size());

    for (auto&& arr : m_stream) { tmp.push_back(arr.size()); }

    return *std::max_element(tmp.begin(), tmp.end());
}
ScalarArray ScalarStream::operator[](dimn_t row) const noexcept
{

    RPY_CHECK(row < m_stream.size());
    return m_stream[row];
}
ScalarCRef ScalarStream::operator[](std::pair<dimn_t, dimn_t> index
) const noexcept
{
    RPY_CHECK(index.first < m_stream.size());
    return m_stream[index.first][index.second];
}
void ScalarStream::set_ctype(TypePtr type) noexcept
{
    RPY_CHECK(m_stream.empty() && p_type == nullptr);
    p_type = std::move(type);
}
void ScalarStream::reserve_size(dimn_t num_rows)
{
    m_stream.reserve(num_rows);
}
void ScalarStream::push_back(const ScalarArray& data)
{
    m_stream.push_back(data);
}
void ScalarStream::push_back(ScalarArray&& data)
{
    m_stream.push_back(std::move(data));
}
