//
// Created by sam on 3/27/24.
//

#ifndef ROUGHPY_ALGEBRA_KEY_SCALAR_STREAM_H
#define ROUGHPY_ALGEBRA_KEY_SCALAR_STREAM_H

#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/scalars_fwd.h>

#include "key_scalar_array.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT KeyScalarStream
{
    containers::SmallVec<KeyScalarArray, 1> m_parts;
    scalars::TypePtr p_type;

public:
    KeyScalarStream() = default;

    explicit KeyScalarStream(scalars::TypePtr type) : p_type(type) {}

    RPY_NO_DISCARD bool empty() const noexcept { return m_parts.empty(); }
    RPY_NO_DISCARD dimn_t col_count(dimn_t i) const noexcept;
    RPY_NO_DISCARD dimn_t max_col_count() const noexcept;
    RPY_NO_DISCARD dimn_t row_count() const noexcept { return m_parts.size(); }

    RPY_NO_DISCARD KeyScalarArray operator[](dimn_t i) const noexcept;
    RPY_NO_DISCARD KeyScalarStream operator[](SliceIndex indices
    ) const noexcept;

    void set_type(scalars::TypePtr type) noexcept;

    void reserve(dimn_t num_rows);

    void push_back(const scalars::ScalarArray& data);
    void push_back(scalars::ScalarArray&& data);
    void push_back(const KeyScalarArray& data);
    void push_back(KeyScalarArray&& data);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_KEY_SCALAR_STREAM_H
