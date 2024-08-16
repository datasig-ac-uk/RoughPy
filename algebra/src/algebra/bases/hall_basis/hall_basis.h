//
// Created by sam on 8/15/24.
//

#ifndef HALL_BASIS_H
#define HALL_BASIS_H

#include "roughpy/algebra/basis.h"
#include <roughpy/devices/core.h>

namespace rpy {
namespace algebra {

class HallBasis : public Basis
{
    deg_t m_width;
    deg_t m_max_degree;
    devices::TypePtr p_lie_word_type;
    devices::TypePtr p_index_key_type;

public:
    HallBasis(deg_t width, deg_t depth);

    RPY_NO_DISCARD bool has_key(BasisKey key) const noexcept override;
    RPY_NO_DISCARD string to_string(BasisKey key) const override;
    RPY_NO_DISCARD bool equals(BasisKey k1, BasisKey k2) const override;
    RPY_NO_DISCARD hash_t hash(BasisKey k1) const override;
    RPY_NO_DISCARD dimn_t max_dimension() const noexcept override;
    RPY_NO_DISCARD dimn_t dense_dimension(dimn_t size) const override;
    RPY_NO_DISCARD bool less(BasisKey k1, BasisKey k2) const override;
    RPY_NO_DISCARD dimn_t to_index(BasisKey key) const override;
    RPY_NO_DISCARD BasisKey to_key(dimn_t index) const override;
    RPY_NO_DISCARD KeyRange iterate_keys() const override;
    RPY_NO_DISCARD dtl::BasisIterator keys_begin() const override;
    RPY_NO_DISCARD dtl::BasisIterator keys_end() const override;
    RPY_NO_DISCARD deg_t max_degree() const override;
    RPY_NO_DISCARD deg_t degree(BasisKey key) const override;
    RPY_NO_DISCARD dimn_t dimension_to_degree(deg_t degree) const override;
    RPY_NO_DISCARD KeyRange iterate_keys_of_degree(deg_t degree) const override;
    RPY_NO_DISCARD deg_t alphabet_size() const override;
    RPY_NO_DISCARD bool is_letter(BasisKey key) const override;
    RPY_NO_DISCARD let_t get_letter(BasisKey key) const override;
    RPY_NO_DISCARD pair<optional<BasisKey>, optional<BasisKey>>
    parents(BasisKey key) const override;
    RPY_NO_DISCARD BasisComparison compare(BasisPointer other
    ) const noexcept override;
    Rc<VectorContext> default_vector_context() const override;
};

}// namespace algebra
}// namespace rpy

#endif// HALL_BASIS_H
