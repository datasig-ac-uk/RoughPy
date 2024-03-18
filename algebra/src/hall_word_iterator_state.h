//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_HALL_WORD_ITERATOR_STATE_H
#define ROUGHPY_HALL_WORD_ITERATOR_STATE_H

#include "basis.h"

namespace rpy {
namespace algebra {

class hall_word_iterator_state : public KeyIteratorState
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

#endif// ROUGHPY_HALL_WORD_ITERATOR_STATE_H
