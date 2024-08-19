//
// Created by sam on 8/15/24.
//

#ifndef HALL_BASIS_H
#define HALL_BASIS_H

#include "roughpy/algebra/basis.h"
#include <roughpy/devices/core.h>

#include <array>

namespace rpy {
namespace algebra {

class HallBasis : public Basis
{
    deg_t m_width;
    deg_t m_max_degree;

    class HallSet;
    Rc<const HallSet> p_hall_set;

    std::array<devices::TypePtr, 2> m_supported_types;

    const devices::TypePtr& lie_word_type() const noexcept
    {
        return m_supported_types[0];
    }

    const devices::TypePtr& index_key_type() const noexcept
    {
        return m_supported_types[1];
    }

    bool is_lie_word(const devices::TypePtr& type) const noexcept
    {
        return type == m_supported_types[0];
    }

    bool is_index_key(const devices::TypePtr& type) const noexcept
    {
        return type == m_supported_types[1];
    }

    optional<dimn_t> key_to_oindex(const BasisKeyCRef& key) const noexcept;

    string to_string_nofail(const BasisKeyCRef& key) const noexcept;

public:
    HallBasis(deg_t width, deg_t depth);


    ~HallBasis() override;

    RPY_NO_DISCARD bool supports_key_type(const devices::TypePtr& type
    ) const noexcept override;
    RPY_NO_DISCARD Slice<const devices::TypePtr>
    supported_key_types() const noexcept override;
    RPY_NO_DISCARD bool has_key(BasisKeyCRef key) const noexcept override;
    RPY_NO_DISCARD string to_string(BasisKeyCRef key) const override;
    RPY_NO_DISCARD bool equals(BasisKeyCRef k1, BasisKeyCRef k2) const override;
    RPY_NO_DISCARD hash_t hash(BasisKeyCRef k1) const override;
    RPY_NO_DISCARD dimn_t max_dimension() const noexcept override;
    RPY_NO_DISCARD dimn_t dense_dimension(dimn_t size) const override;
    RPY_NO_DISCARD bool less(BasisKeyCRef k1, BasisKeyCRef k2) const override;
    RPY_NO_DISCARD dimn_t to_index(BasisKeyCRef key) const override;
    RPY_NO_DISCARD BasisKey to_key(dimn_t index) const override;
    RPY_NO_DISCARD KeyRange iterate_keys() const override;
    RPY_NO_DISCARD dtl::BasisIterator keys_begin() const override;
    RPY_NO_DISCARD dtl::BasisIterator keys_end() const override;
    RPY_NO_DISCARD deg_t max_degree() const override;
    RPY_NO_DISCARD deg_t degree(BasisKeyCRef key) const override;
    RPY_NO_DISCARD deg_t dimension_to_degree(dimn_t dimension) const override;
    RPY_NO_DISCARD KeyRange iterate_keys_of_degree(deg_t degree) const override;
    RPY_NO_DISCARD deg_t alphabet_size() const override;
    RPY_NO_DISCARD bool is_letter(BasisKeyCRef key) const override;
    RPY_NO_DISCARD let_t get_letter(BasisKeyCRef key) const override;
    RPY_NO_DISCARD pair<BasisKey, BasisKey> parents(BasisKeyCRef key
    ) const override;
};

}// namespace algebra
}// namespace rpy

#endif// HALL_BASIS_H
