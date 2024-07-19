//
// Created by sam on 7/8/24.
//

#ifndef VECTOR_DATA_H
#define VECTOR_DATA_H

#include <roughpy/core/smart_ptr.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/alloc.h>

#include "scalars_fwd.h"
#include "scalar_array.h"

namespace rpy {
namespace scalars {
namespace dtl {

class VectorData : public platform::SmallObjectBase, public RcBase<VectorData>
{
    ScalarArray m_scalar_buffer{};
    dimn_t m_size = 0;

public:
    void set_size(dimn_t size)
    {
        RPY_CHECK(size <= m_scalar_buffer.size());
        m_size = size;
    }

    VectorData() = default;

    explicit VectorData(TypePtr type, dimn_t size)
        : m_scalar_buffer(type, size),
          m_size(size)
    {}

    explicit VectorData(ScalarArray&& scalars)
        : m_scalar_buffer(std::move(scalars)),
          m_size(scalars.size())
    {}

    explicit VectorData(TypePtr type) : m_scalar_buffer(type) {}

    void reserve(dimn_t dim);
    void resize(dimn_t dim);

    RPY_NO_DISCARD dimn_t capacity() const noexcept
    {
        return m_scalar_buffer.size();
    }
    RPY_NO_DISCARD dimn_t size() const noexcept { return m_size; }

    RPY_NO_DISCARD bool empty() const noexcept
    {
        return m_scalar_buffer.empty();
    }

    RPY_NO_DISCARD TypePtr scalar_type() const noexcept
    {
        return m_scalar_buffer.type();
    }

    RPY_NO_DISCARD devices::Buffer& mut_scalar_buffer() noexcept
    {
        return m_scalar_buffer.mut_buffer();
    }
    RPY_NO_DISCARD const devices::Buffer& scalar_buffer() const noexcept
    {
        return m_scalar_buffer.buffer();
    }

    RPY_NO_DISCARD ScalarArray& mut_scalars() noexcept
    {
        return m_scalar_buffer;
    }
    RPY_NO_DISCARD const ScalarArray& scalars() const noexcept
    {
        return m_scalar_buffer;
    }

    void insert_element(dimn_t index, dimn_t next_size, Scalar value);
    void delete_element(dimn_t index);
};

}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// VECTOR_DATA_H
