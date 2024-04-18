//
// Created by sam on 2/15/24.
//

#ifndef ROUGHPY_VECTOR_ITERATOR_H
#define ROUGHPY_VECTOR_ITERATOR_H

#include "algebra_fwd.h"
#include "roughpy_algebra_export.h"

#include "basis_key.h"
#include "key_array.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/buffer.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT VectorIterator
{
    scalars::ScalarArray m_scalar_view;
    KeyArray m_key_view;

    dimn_t m_index;

public:
    using value_type = pair<BasisKey, scalars::Scalar>;

    class KVPairProxy
    {
        value_type m_pair;

    public:
        KVPairProxy(BasisKey&& key, scalars::Scalar&& value)
            : m_pair(std::move(key), std::move(value))
        {}

        operator const value_type&() const noexcept { return m_pair; }

        const value_type* operator->() const noexcept { return &m_pair; }
    };

    using reference = KVPairProxy;
    using pointer = KVPairProxy;

    VectorIterator(
            scalars::ScalarArray data_view,
            KeyArray key_view,
            dimn_t index = 0
    )
        : m_scalar_view(std::move(data_view)),
          m_key_view(std::move(key_view)),
          m_index(index)
    {}

    VectorIterator& operator++();

    const VectorIterator operator++(int);

    reference operator*();

    pointer operator->();

    bool operator==(const VectorIterator& other) const noexcept;

    bool operator!=(const VectorIterator& other) const noexcept;
};

}// namespace algebra
}// namespace rpy
#endif// ROUGHPY_VECTOR_ITERATOR_H
