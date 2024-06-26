//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_TENSOR_KEY_ITERATOR_STATE_H
#define ROUGHPY_TENSOR_KEY_ITERATOR_STATE_H

#include "basis.h"
#include "basis_key.h"

namespace rpy {
namespace algebra {

class TensorKeyIteratorState : public KeyIteratorState
{
    dimn_t m_index;
    dimn_t m_max;

public:
    void advance() noexcept override;
    bool finished() const noexcept override;
    BasisKey value() const noexcept override;
    bool equals(BasisKey k1, BasisKey k2) const noexcept override;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_TENSOR_KEY_ITERATOR_STATE_H
