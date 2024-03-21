//
// Created by sam on 3/18/24.
//

#include "tensor_product_basis.h"
#include "tensor_product_basis_key.h"

#include <sstream>

using namespace rpy;
using namespace rpy::algebra;

TensorProductBasis::TensorProductBasis(Slice<BasisPointer> bases)
    : Basis(basis_id, {false, false, false}),
      m_bases(bases)
{}

TensorProductBasis::TensorProductBasis(
        Slice<rpy::algebra::BasisPointer> bases,
        ordering_function order
)
    : Basis(basis_id, {true, false, false}),
      m_bases(bases),
      m_ordering(std::move(order))
{}

bool TensorProductBasis::has_key(BasisKey key) const noexcept
{
    if (key.is_valid_pointer()) {
        const auto* ptr = key.get_pointer();
        if (ptr->key_type() != TensorProductBasisKey::key_name) {
            return false;
        }

        const auto* tpbk = static_cast<const TensorProductBasisKey*>(ptr);
        if (m_bases.size() != tpbk->size()) { return false; }

        auto bit = m_bases.begin();
        auto tpk_it = tpbk->begin();

        for (; bit != m_bases.end(); ++bit, ++tpk_it) {
            if (!(*bit)->has_key(*tpk_it)) { return false; }
        }

        return true;
    }

    if (is_ordered() && key.is_index()) {
        return key.get_index() <= max_dimension();
    }

    return false;
}
string TensorProductBasis::to_string(BasisKey key) const
{
    RPY_CHECK(key.is_valid_pointer());
    const auto* kptr = key.get_pointer();
    RPY_CHECK(kptr->key_type() == TensorProductBasisKey::key_name);
    const auto* tpbk = static_cast<const TensorProductBasisKey*>(kptr);
    RPY_CHECK(tpbk->size() == m_bases.size());

    std::stringstream ss;
    bool first = true;
    ss << '(';

    auto bit = m_bases.begin();
    auto tpk_it = tpbk->begin();

    for (; bit != m_bases.end(); ++bit, ++tpk_it) {
        RPY_CHECK((*bit)->has_key(*tpk_it));
        if (!first) { ss << ','; }
        ss << (*bit)->to_string(*tpk_it);
        first = false;
    }

    ss << ')';
    return ss.str();
}
bool TensorProductBasis::equals(BasisKey k1, BasisKey k2) const
{
    return false;
}
hash_t TensorProductBasis::hash(BasisKey k1) const { return 0; }
bool TensorProductBasis::less(BasisKey k1, BasisKey k2) const
{
    if (m_ordering) {
        RPY_CHECK(has_key(k1) && has_key(k2));
        if (k1.is_index() && k2.is_index()) {
            return k1.get_index() < k2.get_index();
        }
    }
    return Basis::less(k1, k2);
}
dimn_t TensorProductBasis::to_index(BasisKey key) const
{
    return Basis::to_index(key);
}
BasisKey TensorProductBasis::to_key(dimn_t index) const
{
    return Basis::to_key(index);
}
KeyRange TensorProductBasis::iterate_keys() const
{
    return Basis::iterate_keys();
}
deg_t TensorProductBasis::max_degree() const { return Basis::max_degree(); }
deg_t TensorProductBasis::degree(BasisKey key) const
{
    return Basis::degree(key);
}
KeyRange TensorProductBasis::iterate_keys_of_degree(deg_t degree) const
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t TensorProductBasis::alphabet_size() const
{
    return Basis::alphabet_size();
}
bool TensorProductBasis::is_letter(BasisKey key) const
{
    return Basis::is_letter(key);
}
let_t TensorProductBasis::get_letter(BasisKey key) const
{
    return Basis::get_letter(key);
}
pair<optional<BasisKey>, optional<BasisKey>>
TensorProductBasis::parents(BasisKey key) const
{
    return Basis::parents(key);
}
BasisComparison TensorProductBasis::compare(BasisPointer other) const noexcept
{
    if (other == this) { return BasisComparison::IsSame; }

    if (other->id() == basis_id) {
        auto& other_as_product = static_cast<const TensorProductBasis&>(*other);
        if (other_as_product.m_bases.size() != m_bases.size()) {
            return BasisComparison::IsNotCompatible;
        }

        BasisComparison result = BasisComparison::IsSame;
        auto itl = m_bases.begin();
        auto itr = other_as_product.m_bases.begin();
        for (; itl != m_bases.end(); ++itl, ++itr) {
            switch ((*itl)->compare(*itr)) {
                case BasisComparison::IsSame: break;
                case BasisComparison::IsCompatible:
                    result = BasisComparison::IsCompatible;
                    break;
                case BasisComparison::IsNotCompatible:
                    return BasisComparison::IsNotCompatible;
                    break;
            }
        }
        return result;
    }

    return BasisComparison::IsNotCompatible;
}
dimn_t TensorProductBasis::max_dimension() const noexcept
{
    dimn_t dimension = 1;
    for (const auto& basis : m_bases) { dimension *= basis->max_dimension(); }
    return dimension;
}
