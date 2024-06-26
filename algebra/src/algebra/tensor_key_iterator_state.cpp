//
// Created by sam on 3/18/24.
//

#include "tensor_key_iterator_state.h"

using namespace rpy;
using namespace rpy::algebra;

void TensorKeyIteratorState::advance() noexcept { ++m_index; }
bool TensorKeyIteratorState::finished() const noexcept
{
    return m_index >= m_max;
}
BasisKey TensorKeyIteratorState::value() const noexcept
{
    return BasisKey(m_index);
}
bool TensorKeyIteratorState::equals(BasisKey k1, BasisKey k2) const noexcept
{
    RPY_DBG_ASSERT(k1.is_index() && k2.is_index());
    return k1.get_index() == k2.get_index();
}
