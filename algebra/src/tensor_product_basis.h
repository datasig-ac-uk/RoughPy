//
// Created by sam on 3/18/24.
//

#ifndef ROUGHPY_TENSOR_PRODUCT_BASIS_H
#define ROUGHPY_TENSOR_PRODUCT_BASIS_H

#include "basis.h"

#include <roughpy/core/container/vector.h>

namespace rpy {
namespace algebra {

class TensorProductBasis : public Basis
{
    containers::SmallVec<BasisPointer, 2> m_bases;

public:
    using ordering_function = std::function<
            bool(Slice<const BasisPointer>,
                 Slice<const BasisKey>,
                 Slice<const BasisKey>)>;

private:
    ordering_function m_ordering;

public:
    static constexpr string_view basis_id = "tensor_product_basis";

    explicit TensorProductBasis(Slice<BasisPointer> bases);
    explicit
    TensorProductBasis(Slice<BasisPointer> bases, ordering_function order);

    dimn_t max_dimension() const noexcept override;

    bool has_key(BasisKey key) const noexcept override;
    string to_string(BasisKey key) const override;
    bool equals(BasisKey k1, BasisKey k2) const override;
    hash_t hash(BasisKey k1) const override;
    bool less(BasisKey k1, BasisKey k2) const override;
    dimn_t to_index(BasisKey key) const override;
    BasisKey to_key(dimn_t index) const override;
    KeyRange iterate_keys() const override;
    deg_t max_degree() const override;
    deg_t degree(BasisKey key) const override;
    KeyRange iterate_keys_of_degree(deg_t degree) const override;
    deg_t alphabet_size() const override;
    bool is_letter(BasisKey key) const override;
    let_t get_letter(BasisKey key) const override;
    pair<optional<BasisKey>, optional<BasisKey>> parents(BasisKey key
    ) const override;

    BasisComparison compare(BasisPointer other) const noexcept override;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_TENSOR_PRODUCT_BASIS_H
