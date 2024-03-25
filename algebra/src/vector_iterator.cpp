//
// Created by sam on 2/15/24.
//

#include "vector_iterator.h"

#include "basis_key.h"

using namespace rpy;
using namespace rpy::algebra;

using rpy::scalars::Scalar;

VectorIterator& VectorIterator::operator++()
{
    ++m_index;
    return *this;
}

const VectorIterator VectorIterator::operator++(int)
{
    VectorIterator prev(*this);
    ++m_index;
    return prev;
}

namespace {

BasisKey clone_key(const void* kptr)
{
    return *static_cast<const BasisKey*>(kptr);
}

}// namespace

VectorIterator::reference VectorIterator::operator*()
{
    if (m_key_view.empty()) {
        return {BasisKey(m_index), m_scalar_view[m_index]};
    }
    auto* ptr
            = static_cast<const BasisKey*>(m_key_view.buffer().ptr()) + m_index;
    return {clone_key(ptr), m_scalar_view[m_index]};
}

VectorIterator::pointer VectorIterator::operator->()
{
    if (m_key_view.empty()) {
        return {BasisKey(m_index), m_scalar_view[m_index]};
    }
    auto* ptr
            = static_cast<const BasisKey*>(m_key_view.buffer().ptr()) + m_index;
    return {clone_key(ptr), m_scalar_view[m_index]};
}

bool VectorIterator::operator==(const VectorIterator& other) const noexcept
{
    return (m_scalar_view.buffer() == other.m_scalar_view.buffer()
            && m_key_view.buffer() == other.m_key_view.buffer()
            && m_index == other.m_index);
}

bool VectorIterator::operator!=(const VectorIterator& other) const noexcept
{
    return (!(m_scalar_view.buffer() == other.m_scalar_view.buffer())
            || !(m_key_view.buffer() == other.m_key_view.buffer())
            || m_index != other.m_index);
}
