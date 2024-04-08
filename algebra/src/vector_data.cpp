//
// Created by sam on 22/03/24.
//

#include "vector.h"

#include <roughpy/scalars/algorithms.h>
#include "key_algorithms.h"

using namespace rpy;
using namespace rpy::algebra;

void VectorData::reserve(dimn_t dim) {
    if (dim <= capacity()) {
        return;
    }

    auto new_buffer = m_scalar_buffer.type()->allocate(dim);
    scalars::algorithms::copy(new_buffer, m_scalar_buffer);

    if (!m_key_buffer.empty()) {
        KeyArray new_key_buffer(m_key_buffer.device(), dim);
        algorithms::copy(new_key_buffer, m_key_buffer);
        std::swap(m_key_buffer, new_key_buffer);
    }

    std::swap(m_scalar_buffer, new_buffer);
}

void VectorData::resize(dimn_t dim)
{
    reserve(dim);
    m_size = dim;
}

void VectorData::insert_element(
        dimn_t index,
        dimn_t next_size,
        BasisKey key,
        scalars::Scalar value
)
{



}
void VectorData::delete_element(dimn_t index)
{
    RPY_DBG_ASSERT(index < m_scalar_buffer.size());
    auto scalar_view = m_scalar_buffer.mut_view();
    m_size -= 1;
    for (dimn_t i = index; i < m_size; ++i) {
        scalar_view[i] = scalar_view[i + 1];
    }

    if (!m_key_buffer.empty()) {
        auto key_view = m_key_buffer.as_mut_slice();
        for (dimn_t i = index; i < m_size; ++i) {
            key_view[i] = std::move(key_view[i + 1]);
        }
    }
}
