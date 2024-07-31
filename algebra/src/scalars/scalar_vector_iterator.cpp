//
// Created by sam on 25/06/24.
//

#include "scalar_vector.h"

using namespace rpy;
using namespace rpy::scalars;

using namespace rpy::scalars::dtl;

ScalarVectorIterator::ScalarVectorIterator() : m_data(), m_index(0) {}

ScalarVectorIterator::ScalarVectorIterator(const ScalarVectorIterator& other)
    : m_data(other.m_data),
      m_index(other.m_index)
{}

ScalarVectorIterator::ScalarVectorIterator(ScalarVectorIterator&& other
) noexcept
    : m_data(std::move(other.m_data)),
      m_index(other.m_index)
{}

ScalarVectorIterator&
ScalarVectorIterator::operator=(const ScalarVectorIterator& other)
{
    if (&other != this) {
        m_data = other.m_data;
        m_index = other.m_index;
    }
    return *this;
}

ScalarVectorIterator& ScalarVectorIterator::operator=(ScalarVectorIterator&& other) noexcept
{
    if (&other != this) {
        m_data = std::move(other.m_data);
        m_index = other.m_index;
    }
    return *this;
}

ScalarVectorIterator& ScalarVectorIterator::operator++()
{
    ++m_index;
    return *this;
}

const ScalarVectorIterator ScalarVectorIterator::operator++(int)
{
    ScalarVectorIterator tmp(*this);
    operator++();
    return tmp;
}

ScalarCRef ScalarVectorIterator::operator*() const
{
    return m_data[m_index];
}

ScalarCRefPointerProxy ScalarVectorIterator::operator->() const
{
    return m_data[m_index];
}


bool scalars::dtl::operator==(const ScalarVectorIterator& lhs, const ScalarVectorIterator& rhs)
{
    if (rhs.m_data.is_null()) {
        return lhs.m_index < lhs.m_data.size();
    }

    if (lhs.m_data.is_null()) {
        return rhs.m_index < rhs.m_data.size();
    }

    if (lhs.m_data == rhs.m_data) {
        return lhs.m_index == rhs.m_index;
    }

    return false;
}