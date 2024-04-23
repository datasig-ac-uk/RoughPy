//
// Created by sam on 3/18/24.
//

#include "tensor_product_basis.h"
#include "tensor_product_basis_key.h"

#include <roughpy/core/ranges.h>

#include <sstream>

using namespace rpy;
using namespace rpy::algebra;

TensorProductBasis::TensorProductBasis(Slice<BasisPointer> bases)
    : Basis(basis_id, {false, false, false})
{
    m_bases.reserve(bases.size());
    for (auto&& basis : bases | views::move) {
        m_bases.emplace_back(std::move(basis));
    }
}

TensorProductBasis::TensorProductBasis(
        Slice<rpy::algebra::BasisPointer> bases,
        ordering_function order
)
    : Basis(basis_id, {true, false, false}),
      m_ordering(std::move(order))
{
    m_bases.reserve(bases.size());
    for (auto&& basis : bases | views::move) {
        m_bases.emplace_back(std::move(basis));
    }
}

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
    if (k1.is_index() && k2.is_index()) {
        return k1.get_index() == k2.get_index();
    }

    auto do_equals = [this](const BasisKey& _k1, const BasisKey& _k2) {
        auto* tk1 = cast_key<TensorProductBasisKey>(_k1);
        auto* tk2 = cast_key<TensorProductBasisKey>(_k2);

        return ranges::fold_left(
                views::zip(m_bases, tk1->keys(), tk2->keys())
                        | views::transform([](const auto& t) {
                              auto&& [b, k1i, k2i] = t;
                              return b->equals(k1i, k2i);
                          }),
                true,
                std::logical_and<>()
        );
    };

    if (k1.is_index()) { return do_equals(to_key(k1.get_index()), k2); }
    if (k2.is_index()) { return do_equals(k1, to_key(k2.get_index())); }
    return do_equals(k1, k2);
}
hash_t TensorProductBasis::hash(BasisKey k1) const
{
    if (k1.is_index()) { return static_cast<hash_t>(k1.get_index()); }

    auto* this_key = cast_key<TensorProductBasisKey>(k1);

    hash_t result = 0;
    for (const auto& [basis, key] : views::zip(m_bases, this_key->keys())) {
        hash_combine(result, basis->hash(key));
    }
    return result;
}
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
deg_t TensorProductBasis::max_degree() const
{
    return ranges::fold_left(
            m_bases | views::transform([](const auto& b) {
                return b->max_degree();
            }),
            0,
            std::plus<>()
    );
}
deg_t TensorProductBasis::degree(BasisKey key) const
{

    auto* this_key = cast_key<TensorProductBasisKey>(key);

    deg_t result = 0;
    for (const auto& [key, basis] : views::zip(this_key->keys(), m_bases)) {
        result += basis->degree(key);
    }
    return result;
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

BasisPointer algebra::tensor_product_basis(
        Slice<BasisPointer> bases,
        std::function<dimn_t(BasisKey)> index_function,
        std::function<BasisKey(dimn_t)> key_function
)
{
    RPY_CHECK(!bases.empty());

    if (bases.size() == 1) { return bases[0]; }

    return BasisPointer(new TensorProductBasis(bases));
}
