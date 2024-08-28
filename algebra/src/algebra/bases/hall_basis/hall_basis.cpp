//
// Created by sam on 8/15/24.
//

#include "bases/hall_basis.h"

#include "index_key_type.h"
#include "lie_word.h"
#include "lie_word_type.h"

#include <roughpy/core/hash.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/sync.h>

#include <roughpy/core/container/map.h>
#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/container/vector.h>

#include <roughpy/devices/type.h>

#include <memory>

#include "hall_set_size.h"

using namespace rpy;
using namespace rpy::algebra;

class HallBasis::HallSet
{
    using parent_type = pair<dimn_t, dimn_t>;

    containers::FlatMap<parent_type, dimn_t> m_reverse_map;
    containers::FlatMap<dimn_t, let_t> m_letters;
    containers::Vec<parent_type> m_elements;
    containers::Vec<dimn_t> m_degree_sizes;
    containers::Vec<pair<dimn_t, dimn_t>> m_degree_ranges;

    template <typename LetterFn, typename Binop>
    decltype(auto)
    do_foliage_map(dimn_t index, const LetterFn& letter_fn, const Binop& binop)
            const
    {
        RPY_DBG_ASSERT(index > 0);
        // We have already checked that index belongs to the set.
        const auto& pars = m_elements[index];
        if (pars.first == 0) {
            const auto it = m_letters.find(pars.second);
            RPY_DBG_ASSERT(it != m_letters.end());
            return letter_fn(it->second);
        }
        return binop(
                do_foliage_map(pars.first, letter_fn, binop),
                do_foliage_map(pars.second, letter_fn, binop)
        );
    }

    bool check_index(dimn_t index) const noexcept
    {
        return index < m_elements.size();
    }

public:
    HallSet(deg_t width, deg_t depth);

    HallSet(const HallSet& other, deg_t depth);

    deg_t width() const noexcept
    {
        return static_cast<deg_t>(m_degree_sizes[1]);
    }

    dimn_t size(deg_t degree) const noexcept
    {
        return degree > m_degree_sizes.size() ? m_degree_sizes.back()
                                              : m_degree_sizes[degree];
    }

    bool is_valid_letter(let_t letter) const noexcept
    {
        return ranges::contains(m_letters | views::values, letter);
    }

    deg_t max_degree() const noexcept { return m_degree_sizes.size() - 1; }
    dimn_t letter_to_index(let_t letter) const noexcept
    {
        const auto it = ranges::find(m_letters, letter, [](const auto& p) {
            return p.second;
        });
        RPY_DBG_ASSERT(it != m_letters.end());
        return it->first;
    }
    let_t index_to_letter(dimn_t index) const
    {
        const auto it = m_letters.find(index);
        RPY_CHECK(it != m_letters.end());
        return it->second;
    }
    optional<dimn_t> pair_to_index(dimn_t left, dimn_t right) const noexcept
    {
        const auto found = m_reverse_map.find({left, right});
        if (found != m_reverse_map.end()) { return found->second; }
        return {};
    }

    Slice<const dimn_t> sizes() const noexcept
    {
        return {m_degree_sizes.data(), m_degree_sizes.size()};
    }

    pair<dimn_t, dimn_t> parents(dimn_t index) const noexcept
    {
        RPY_DBG_ASSERT(index < m_elements.size());
        return m_elements[index];
    }

    template <typename LetterFn, typename Binop>
    decltype(auto)
    foliage_map(dimn_t index, LetterFn&& letter_fn, Binop&& binop) const
    {
        RPY_CHECK(check_index(index));
        if (is_letter(index)) { return letter_fn(index); }
        return do_foliage_map(index, letter_fn, binop);
    }

    deg_t degree(dimn_t index) const noexcept
    {
        const auto begin = m_degree_sizes.begin();
        const auto end = m_degree_sizes.end();
        auto it = ranges::lower_bound(begin + 1, end, index, std::less_equal{});

        return static_cast<deg_t>(it - begin);
    }

    bool is_letter(dimn_t index) const noexcept
    {
        return m_elements[index].first == 0;
    }

    pair<dimn_t, dimn_t> range_of_degree(deg_t degree) const noexcept
    {
        RPY_DBG_ASSERT(degree < m_degree_ranges.size());
        return m_degree_ranges[degree];
    }

private:
    void grow(deg_t depth);

public:
    static std::shared_ptr<const HallSet> get(deg_t width, deg_t depth);
};

HallBasis::HallSet::HallSet(deg_t width, deg_t depth)
{
    m_letters.reserve(width);

    HallSetSizeHelper helper(width);
    const auto hs_size = helper(std::max(depth, 1));

    m_elements.reserve(hs_size + 1);
    m_elements.emplace_back(0, 0);
    m_reverse_map.reserve(hs_size);

    m_degree_ranges.reserve(1 + depth);
    m_degree_ranges.emplace_back(0, 0);

    m_degree_sizes.reserve(1 + depth);
    m_degree_sizes.emplace_back(1);

    for (let_t l = 1; l <= static_cast<let_t>(width); ++l) {
        dimn_t index = l;
        parent_type parents{0, l};

        m_letters.emplace(index, l);
        m_elements.emplace_back(parents);
        m_reverse_map.emplace(parents, index);
    }

    m_degree_sizes.emplace_back(1 + width);
    m_degree_ranges.emplace_back(1, m_elements.size());

    grow(depth);
}

void HallBasis::HallSet::grow(deg_t depth)
{
    auto degree = max_degree();

    if (degree >= depth) { return; }

    for (auto d = degree + 1; d <= depth; ++d) {
        dimn_t index = m_elements.size();

        for (deg_t e = 1; 2 * e <= d; ++e) {
            auto i_bounds = m_degree_ranges[e];
            auto j_bounds = m_degree_ranges[d - e];

            for (auto i = i_bounds.first; i < i_bounds.second; ++i) {
                const auto jmin = std::max(j_bounds.first, i + 1);

                for (auto j = jmin; j < j_bounds.second; ++j) {
                    if (m_elements[j].first <= i) {
                        dimn_t key = index++;
                        parent_type parents{i, j};

                        m_elements.emplace_back(parents);
                        m_reverse_map.emplace(parents, key);
                    }
                }
                RPY_DBG_ASSERT(m_elements.size() == index);
            }
        }

        m_degree_ranges.emplace_back(
                m_degree_ranges.back().second,
                m_elements.size()
        );
        m_degree_sizes.push_back(m_elements.size());
    }
}

HallBasis::HallSet::HallSet(const HallSet& other, deg_t depth)
    : m_letters(other.m_letters)
{
    HallSetSizeHelper helper(other.width());
    RPY_DBG_ASSERT(depth > other.max_degree());

    const auto hs_size = helper(depth);

    m_elements.reserve(hs_size);
    m_elements.insert(
            m_elements.end(),
            other.m_elements.begin(),
            other.m_elements.end()
    );
    m_reverse_map.reserve(hs_size);
    m_reverse_map.insert(
            other.m_reverse_map.begin(),
            other.m_reverse_map.end()
    );

    m_degree_ranges.reserve(depth);
    m_degree_ranges.insert(
            m_degree_ranges.end(),
            other.m_degree_ranges.begin(),
            other.m_degree_ranges.end()
    );
    m_degree_sizes.reserve(depth);
    m_degree_sizes.insert(
            m_degree_sizes.end(),
            other.m_degree_sizes.begin(),
            other.m_degree_sizes.end()
    );

    grow(depth);
}

std::shared_ptr<const typename HallBasis::HallSet>
HallBasis::HallSet::get(deg_t width, deg_t depth)
{
    static Mutex lock;
    static containers::HashMap<deg_t, std::shared_ptr<HallSet>> cache;

    LockGuard<Mutex> access(lock);
    auto& entry = cache[width];
    if (entry) {
        if (entry->max_degree() < depth) {
            // Entry exists but is not big enough.
            // Copy the old one, and grow until it is the correct size
            auto new_hallset = std::make_shared<HallSet>(*entry, depth);

            // Swap the old cached Hall set with the new. This way bases that
            // use the old set will continue to use them. New bases will use the
            // existing one
            std::swap(entry, new_hallset);
        }
    } else {
        // The entry does not exist. Create it
        entry = std::make_shared<HallSet>(width, depth);
    }

    return entry;
}

namespace {

constexpr bool is_hs_null(dimn_t index) noexcept { return index == 0; }
constexpr dimn_t to_hs_index(dimn_t index) noexcept { return index + 1; }
constexpr dimn_t from_hs_index(dimn_t index) noexcept
{
    RPY_DBG_ASSERT(!is_hs_null(index));
    return index - 1;
}

constexpr dimn_t to_hs_dim(dimn_t dim) noexcept { return dim + 1; }
constexpr dimn_t adjust_hs_dim(dimn_t dim) noexcept { return dim - 1; }
constexpr bool check_hs_index(dimn_t index, dimn_t dim) noexcept
{
    return to_hs_index(index) < dim;
}

}// namespace

/* -----------------------------------------------------------------------------
 * Implementation of Hall Basis
 * -----------------------------------------------------------------------------
 */

HallBasis::HallBasis(deg_t width, deg_t depth)
    : Basis("hall_basis", {true, true, true}),
      p_hall_set(HallSet::get(width, depth)),
      m_width(width),
      m_max_degree(depth),
      m_supported_types{LieWordType::get(), IndexKeyType::get()}
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

inline const LieWord* cast_word(const BasisKeyCRef& key) noexcept
{
    RPY_DBG_ASSERT(key.type()->id() == "lie_word");
    return key.data<LieWord>();
}

inline const dimn_t cast_index(const BasisKeyCRef& key) noexcept
{
    RPY_DBG_ASSERT(key.type()->id() == "index_key");
    return key.value<dimn_t>();
}

}// namespace

algebra::dtl::BasisIterator HallBasis::make_iterator(dimn_t index
) const noexcept
{
    return dtl::BasisIterator(
            +[](BasisKeyCRef current) {
                return BasisKey(current.type(), cast_index(current) + 1);
            },
            BasisKey(index_key_type(), index)
    );
}
optional<dimn_t> HallBasis::key_to_oindex(const BasisKeyCRef& key
) const noexcept
{
    auto* key_ptr = cast_word(key);
    if (key_ptr->is_letter()) {
        try {
            return get_letter(key);
        } catch (...) {
            return {};
        }
    }

    return cast_word(key)->foliage_map(
            [this](let_t letter) {
                return optional<dimn_t>(p_hall_set->letter_to_index(letter));
            },
            [this](optional<dimn_t> left, optional<dimn_t> right) {
                if (!left || !right) { return left; }

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

    const auto& tp = key.type();
    if (tp) { return string_cat("of type ", tp->name()); }

    return "undefined type";
}

bool HallBasis::has_key(BasisKeyCRef key) const noexcept
{
    if (is_lie_word(key.type())) {
        return static_cast<bool>(key_to_oindex(key));
    }
    if (is_index_key(key.type())) {
        return check_hs_index(cast_index(key), p_hall_set->size(m_max_degree));
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
                to_hs_index(cast_index(key)),
                [](let_t letter) { return std::to_string(letter); },
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
                to_hs_index(cast_index(k1)),
                LieWord::letter_hash,
                LieWord::hash_binop
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
    return adjust_hs_dim(p_hall_set->size(m_max_degree));
}
dimn_t HallBasis::dense_dimension(dimn_t size) const
{
    const auto sizes = p_hall_set->sizes();
    const auto begin = sizes.begin();
    const auto end = sizes.end();
    auto pos = ranges::lower_bound(
            begin,
            end,
            to_hs_dim(size),
            std::less_equal{}
    );
    RPY_CHECK(pos != end);
    return adjust_hs_dim(*pos);
}
bool HallBasis::less(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    return to_index(k1) < to_index(k2);
}
dimn_t HallBasis::to_index(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) {
        auto oindex = key_to_oindex(key);
        RPY_CHECK(oindex);
        return from_hs_index(*oindex);
    }
    if (is_index_key(key.type())) {
        auto index = cast_index(key);
        RPY_CHECK(check_hs_index(index, p_hall_set->size(m_max_degree)));
        return index;
    }

    RPY_THROW(
            std::runtime_error,
            string_cat("unsupported key type ", key.type()->name())
    );
}
BasisKey HallBasis::to_key(dimn_t index) const
{
    RPY_CHECK(check_hs_index(index, p_hall_set->size(m_max_degree)));

    return BasisKey(p_hall_set->foliage_map(
            to_hs_index(index),
            [](let_t letter) { return LieWord(letter); },
            [](const LieWord& left, const LieWord& right) {
                return left * right;
            }
    ));
}
KeyRange HallBasis::iterate_keys() const
{
    return KeyRange(keys_begin(), keys_end());
}
algebra::dtl::BasisIterator HallBasis::keys_begin() const
{
    return make_iterator(0);
}
algebra::dtl::BasisIterator HallBasis::keys_end() const
{
    return make_iterator(max_dimension());
}
deg_t HallBasis::max_degree() const { return m_max_degree; }
deg_t HallBasis::degree(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) { return cast_word(key)->degree(); }
    if (is_index_key(key.type())) {
        return p_hall_set->degree(to_hs_index(cast_index(key)));
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

    /*
     * The size array as provided by the HallSet is of the form
     *
     *  { 1, 1 + width, ... }
     *
     *  which means we need to look for the adjusted dimension.
     */

    auto pos = ranges::lower_bound(
            begin + 1,
            end,
            to_hs_dim(dimension),
            std::less_equal{}
    );
    return static_cast<deg_t>(pos - begin);
}
dimn_t HallBasis::degree_to_dimension(deg_t degree) const
{
    RPY_CHECK(degree <= m_max_degree);

    return adjust_hs_dim(p_hall_set->size(degree));
}
KeyRange HallBasis::iterate_keys_of_degree(deg_t degree) const
{
    RPY_CHECK(degree <= m_max_degree);
    const auto range = p_hall_set->range_of_degree(degree);
    return KeyRange(
            make_iterator(from_hs_index(range.first)),
            make_iterator(from_hs_index(range.second))
    );
}
deg_t HallBasis::alphabet_size() const { return m_width; }
bool HallBasis::is_letter(BasisKeyCRef key) const
{
    if (is_lie_word(key.type())) {
        if (cast_word(key)->is_letter()) {
            const auto letter = cast_word(key)->get_letter();
            RPY_CHECK(p_hall_set->is_valid_letter(letter));
            return true;
        }
        return false;
    }
    if (is_index_key(key.type())) {
        auto index = cast_index(key);
        RPY_CHECK(check_hs_index(index, p_hall_set->size(m_max_degree)));
        return p_hall_set->is_letter(to_hs_index(index));
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
let_t HallBasis::get_letter(BasisKeyCRef key) const
{
    RPY_CHECK(is_letter(key));
    if (is_index_key(key.type())) {
        return p_hall_set->index_to_letter(to_hs_index(cast_index(key)));
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
        if (check_hs_index(index, p_hall_set->size(m_max_degree))) {
            const auto parents = p_hall_set->parents(to_hs_dim(index));

            pair<BasisKey, BasisKey> result;

            // The first element is zero if the key is a letter.
            if (!is_hs_null(parents.first)) {
                result.first = BasisKey(
                        index_key_type(),
                        from_hs_index(parents.first)
                );
            }

            // The second element is always valid
            result.second
                    = BasisKey(index_key_type(), from_hs_index(parents.second));

            return result;
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

Rc<const HallBasis> HallBasis::get(deg_t width, deg_t depth)
{
    return new HallBasis(width, depth);
}
