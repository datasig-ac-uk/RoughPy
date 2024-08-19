//
// Created by sam on 8/15/24.
//

#include "hall_basis.h"

#include "lie_word.h"
#include "lie_word_type.h"

#include <roughpy/core/hash.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>

using namespace rpy;
using namespace rpy::algebra;

class HallBasis::HallSet : public RcBase<HallSet>
{

public:
    dimn_t size(deg_t degree) const noexcept;
    dimn_t letter_to_index(let_t letter) const noexcept;
    let_t index_to_letter(dimn_t index) const noexcept;
    optional<dimn_t> pair_to_index(dimn_t left, dimn_t right) const noexcept;

    Slice<const dimn_t> sizes() const noexcept;

    pair<dimn_t, dimn_t> parents(dimn_t index) const noexcept;

    template <typename LetterFn, typename Binop>
    decltype(auto)
    foliage_map(dimn_t index, LetterFn&& letter_fn, Binop&& binop) const;
};

HallBasis::HallBasis(deg_t width, deg_t depth)
    : Basis("hall_basis", {true, true, true}),
      m_width(width),
      m_max_degree(depth),
      m_supported_types{LieWordType::get(), nullptr}
{}

HallBasis::~HallBasis() = default;

bool HallBasis::supports_key_type(const devices::TypePtr& type) const noexcept
{
    return type == lie_word_type() || type == index_key_type();
}
Slice<const devices::TypePtr> HallBasis::supported_key_types() const noexcept
{
    return {m_supported_types.data(), m_supported_types.size()};
}

namespace {

constexpr const LieWord* cast_word(const BasisKeyCRef& key) noexcept
{
    RPY_DBG_ASSERT(key.type()->id() == "lie_word");
    return key.data<LieWord>();
}

constexpr const dimn_t cast_index(const BasisKeyCRef& key) noexcept
{
    RPY_DBG_ASSERT(key.type()->id() == "index_key");
    return key.value<dimn_t>();
}

}// namespace

optional<dimn_t> HallBasis::key_to_oindex(const BasisKeyCRef& key
) const noexcept
{
    return cast_word(key)->foliage_map(
            [this](let_t letter) {
                return optional<dimn_t>(p_hall_set->letter_to_index(letter));
            },
            [this](optional<dimn_t> left, optional<dimn_t> right) {
                if (!left || !right) { return {}; }

                return p_hall_set->pair_to_index(*left, *right);
            }
    );
}

string HallBasis::to_string_nofail(const BasisKeyCRef& key) const noexcept
{
    if (is_index_key(key.type())) { return std::to_string(cast_index(key)); }
    if (is_lie_word(key.type())) {
        std::stringstream ss;
        cast_word(key)->print(ss);
        return ss.str();
    }

    return "";
}

bool HallBasis::has_key(BasisKeyCRef key) const noexcept
{
    if (is_lie_word(key.type())) {
        return static_cast<bool>(key_to_oindex(key));
    }
    if (is_index_key(key.type())) {
        return cast_index(key) < p_hall_set->size(m_max_degree);
    }
    return false;
}
string HallBasis::to_string(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) {
        std::stringstream ss;
        cast_word(key)->print(ss);
        return ss.str();
    }
    if (is_index_key(key.type())) {
        return p_hall_set->foliage_map(
                cast_index(key),
                [](let_t letter) { return std::toupper(letter); },
                [](const string& left, const string& right) {
                    return string_cat('[', left, ',', right, ']');
                }
        );
    }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "key ",
                    to_string_nofail(key),
                    " does not belong to this basis"
            )
    );
}
bool HallBasis::equals(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    const auto lindex = to_index(k1);
    const auto rindex = to_index(k2);
    return lindex == rindex;
}
hash_t HallBasis::hash(BasisKeyCRef k1) const
{
    if (is_lie_word(k1.type())) { return hash_value(*cast_word(k1)); }
    if (is_index_key(k1.type())) {
        return p_hall_set->foliage_map(
                cast_index(k1),
                [](let_t letter) {
                    Hash<let_t> hasher;
                    return hasher(letter);
                },
                [](hash_t left, hash_t right) {
                    return hash_combine(left, right);
                }
        );
    }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "key ",
                    to_string_nofail(k1),
                    " does not belong to this basis"
            )
    );
}
dimn_t HallBasis::max_dimension() const noexcept
{
    return p_hall_set->size(m_max_degree);
}
dimn_t HallBasis::dense_dimension(dimn_t size) const
{
    const auto sizes = p_hall_set->sizes();
    const auto begin = sizes.begin();
    const auto end = sizes.end();
    auto pos = ranges::lower_bound(begin, end, size);
    RPY_CHECK(pos != end);
    return *pos;
}
bool HallBasis::less(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    return Basis::less(k1, k2);
}
dimn_t HallBasis::to_index(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) {
        auto oindex = key_to_oindex(key);
        RPY_CHECK(oindex);
        return *oindex;
    }
    if (is_index_key(key.type())) {
        auto index = cast_index(key);
        RPY_CHECK(index < p_hall_set->size(m_max_degree));
        return index;
    }


}
BasisKey HallBasis::to_key(dimn_t index) const
{
    RPY_CHECK(index < p_hall_set->size(m_max_degree));

    return BasisKey(p_hall_set->foliage_map(
            index,
            [](let_t letter) { return LieWord(letter); },
            [](const LieWord& left, const LieWord& right) {
                return left * right;
            }
    ));
}
KeyRange HallBasis::iterate_keys() const { return Basis::iterate_keys(); }
algebra::dtl::BasisIterator HallBasis::keys_begin() const
{
    return Basis::keys_begin();
}
algebra::dtl::BasisIterator HallBasis::keys_end() const
{
    return Basis::keys_end();
}
deg_t HallBasis::max_degree() const { return m_max_degree; }
deg_t HallBasis::degree(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) { return cast_word(key)->degree(); }
    if (is_index_key(key.type())) {
        return HallBasis::dimension_to_degree(cast_index(key));
    }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "key ",
                    to_string_nofail(key),
                    " does not belong to this basis"
            )
    );
}

deg_t HallBasis::dimension_to_degree(dimn_t dimension) const
{
    const auto sizes = p_hall_set->sizes();
    const auto begin = sizes.begin();
    const auto end = sizes.end();

    auto pos = ranges::lower_bound(begin, end, dimension);

    RPY_CHECK(pos != begin);

    return static_cast<deg_t>((--pos) - begin);
}
KeyRange HallBasis::iterate_keys_of_degree(deg_t degree) const
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t HallBasis::alphabet_size() const { return Basis::alphabet_size(); }
bool HallBasis::is_letter(BasisKeyCRef key) const
{
    return HallBasis::degree(key) == 1;
}
let_t HallBasis::get_letter(BasisKeyCRef key) const
{
    RPY_CHECK(is_letter(key));
    if (is_index_key(key.type())) {
        return p_hall_set->index_to_letter(cast_index(key));
    }

    if (is_lie_word(key.type())) { return cast_word(key)->get_letter(); }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "key ",
                    to_string_nofail(key),
                    " does not belong to this basis"
            )
    );
}
pair<BasisKey, BasisKey> HallBasis::parents(BasisKeyCRef key) const
{
    RPY_CHECK(has_key(key));
    if (is_lie_word(key.type())) {
        const auto* lkey = cast_word(key);
        return {BasisKey(lkey->left_parent()), BasisKey(lkey->right_parent())};
    }

    if (is_index_key(key.type())) {
        const auto index = cast_index(key);
        if (index < p_hall_set->size(m_max_degree)) {
            const auto parents = p_hall_set->parents(index);
            return {BasisKey(index_key_type(), parents.first),
                    BasisKey(index_key_type(), parents.second)};
        }
    }

    RPY_THROW(
            std::runtime_error,
            string_cat(
                    "key ",
                    to_string_nofail(key),
                    " does not belong to this basis"
            )
    );
}
