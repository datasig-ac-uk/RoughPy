//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_LIE_BASIS_H
#define ROUGHPY_ALGEBRA_LIE_BASIS_H

#include "basis.h"
#include "basis_key.h"
#include "roughpy_algebra_export.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT LieBasis : public Basis
{
    deg_t m_width;
    deg_t m_depth;
    dimn_t m_max_dimension;

    class HallSet;

    std::shared_ptr<HallSet> p_hallset;

public:
    using parent_type = pair<BasisKey, BasisKey>;

private:
public:
    using key_type = BasisKey;

    LieBasis(deg_t width, deg_t depth);

    bool has_key(BasisKey key) const noexcept override;
    string to_string(BasisKey key) const noexcept override;
    bool equals(BasisKey k1, BasisKey k2) const noexcept override;
    hash_t hash(BasisKey k1) const noexcept override;
    bool less(BasisKey k1, BasisKey k2) const noexcept override;
    dimn_t to_index(BasisKey key) const override;
    BasisKey to_key(dimn_t index) const override;
    KeyRange iterate_keys() const noexcept override;
    deg_t max_degree() const noexcept override;
    deg_t degree(BasisKey key) const noexcept override;
    KeyRange iterate_keys_of_degree(deg_t degree) const noexcept override;
    deg_t alphabet_size() const noexcept override;
    bool is_letter(BasisKey key) const noexcept override;
    let_t get_letter(BasisKey key) const noexcept override;
    pair<optional<BasisKey>, optional<BasisKey>> parents(BasisKey key
    ) const noexcept override;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LIE_BASIS_H
