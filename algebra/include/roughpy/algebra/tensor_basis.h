//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H

#include "basis.h"

#include <roughpy/core/container/vector.h>
#include <roughpy/devices/buffer.h>

namespace rpy {
namespace algebra {

/**
 * @class TensorBasis
 * @brief A class representing a tensor basis.
 *
 * The TensorBasis class is a concrete implementation of the Basis abstract
 * class. It provides functionality for manipulating and working with a tensor
 * basis.
 */
class ROUGHPY_ALGEBRA_EXPORT TensorBasis : public Basis
{
    deg_t m_width;
    deg_t m_depth;
    dimn_t m_max_dimension;

    containers::Vec<dimn_t> m_degree_sizes;

    TensorBasis(deg_t width, deg_t depth);

public:
    static constexpr string_view basis_id = "tensor_basis";

    dimn_t max_dimension() const noexcept override;
    RPY_NO_DISCARD dimn_t dense_dimension(dimn_t size) const override;
    bool has_key(BasisKey key) const noexcept override;
    string to_string(BasisKey key) const override;
    bool equals(BasisKey k1, BasisKey k2) const override;
    hash_t hash(BasisKey k1) const override;

    bool less(BasisKey k1, BasisKey k2) const override;
    dimn_t to_index(BasisKey key) const override;
    BasisKey to_key(dimn_t index) const override;
    KeyRange iterate_keys() const override;
    deg_t max_degree() const noexcept override;
    deg_t degree(BasisKey key) const override;
    KeyRange iterate_keys_of_degree(deg_t degree) const override;
    deg_t alphabet_size() const noexcept override;
    bool is_letter(BasisKey key) const override;
    let_t get_letter(BasisKey key) const override;
    pair<optional<BasisKey>, optional<BasisKey>> parents(BasisKey key
    ) const override;

    BasisComparison compare(BasisPointer other) const noexcept override;

    static BasisPointer get(deg_t width, deg_t depth);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H
