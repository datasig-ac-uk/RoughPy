//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H

#include "basis.h"
#include "basis_key.h"


#include <roughpy/platform/devices/buffer.h>

namespace rpy { namespace algebra {


class TensorBasis : public Basis {
    deg_t m_width;
    deg_t m_depth;
    dimn_t m_max_dimension;

    std::vector<dimn_t> m_degree_sizes;

public:

    TensorBasis(deg_t width, deg_t depth);

    virtual bool has_key(BasisKey key) const noexcept;
    virtual string to_string(BasisKey key) const noexcept;
    virtual bool equals(BasisKey k1, BasisKey k2) const noexcept;
    virtual hash_t hash(BasisKey k1) const noexcept;

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


}}


#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H
