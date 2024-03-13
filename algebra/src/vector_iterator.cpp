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

}


VectorIterator::reference VectorIterator::operator*()
{
    if (m_key_view.empty()) {
        return {BasisKey(m_index), m_scalar_view[m_index]};
    }
    return {clone_key(m_key_view.raw_ptr(m_index * sizeof(BasisKey))),
            m_scalar_view[m_index]};
}

VectorIterator::pointer VectorIterator::operator->()
{
    if (m_key_view.empty()) {
        return {BasisKey(m_index), m_scalar_view[m_index]};
    }
    return {clone_key(m_key_view.raw_ptr(m_index * sizeof(BasisKey))),
            m_scalar_view[m_index]};
}

bool VectorIterator::operator==(const VectorIterator& other) const noexcept
{
    return (m_scalar_view.memory_view() == other.m_scalar_view.memory_view()
            && m_key_view == other.m_key_view && m_index == other.m_index);
}

bool VectorIterator::operator!=(const VectorIterator& other) const noexcept
{
    return (!(m_scalar_view.memory_view() == other.m_scalar_view.memory_view())
            || !(m_key_view == other.m_key_view) || m_index != other.m_index);
}
