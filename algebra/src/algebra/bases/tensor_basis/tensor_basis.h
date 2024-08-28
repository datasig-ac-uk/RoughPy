//
// Created by sam on 16/02/24.
//

#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H

#include <roughpy/algebra/basis.h>

#include <roughpy/core/container/vector.h>
#include <roughpy/devices/buffer.h>

#include <memory>

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
    Slice<const dimn_t> m_degree_sizes;
    std::array<KeyTypePtr, 2> m_supported_key_types;

    class Details;
    std::shared_ptr<const Details> p_details;

    deg_t m_width;
    deg_t m_depth;

    TensorBasis(deg_t width, deg_t depth);

    bool is_word(const BasisKeyCRef& key) const noexcept;
    bool is_index(const BasisKeyCRef& key) const noexcept;

    string to_string_nofail(const BasisKeyCRef& key) const noexcept;

    const devices::Type* index_word_type() const noexcept
    {
        return m_supported_key_types[0].get();
    }
    const devices::Type* index_key_type() const noexcept
    {
        return m_supported_key_types[1].get();
    }

    dtl::BasisIterator make_iterator(dimn_t index) const noexcept;

    KeyTypePtr word_type() const noexcept { return m_supported_key_types[0]; }
    KeyTypePtr index_type() const noexcept { return m_supported_key_types[1]; }

public:
    static constexpr string_view basis_id = "tensor_basis";
    BasisComparison compare(BasisPointer other) const noexcept override;

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
    RPY_NO_DISCARD dimn_t degree_to_dimension(deg_t degree) const override;
    RPY_NO_DISCARD KeyRange iterate_keys_of_degree(deg_t degree) const override;
    RPY_NO_DISCARD deg_t alphabet_size() const override;
    RPY_NO_DISCARD bool is_letter(BasisKeyCRef key) const override;
    RPY_NO_DISCARD let_t get_letter(BasisKeyCRef key) const override;
    RPY_NO_DISCARD pair<BasisKey, BasisKey> parents(BasisKeyCRef key
    ) const override;

    static BasisPointer get(deg_t width, deg_t depth);
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H
