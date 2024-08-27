//
// Created by sam on 16/02/24.
//

#include "tensor_basis.h"

#include <roughpy/core/hash.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "index_key_type.h"
#include "tensor_word.h"
#include "tensor_word_type.h"
#include "to_letter_iterator.h"

using namespace rpy;
using namespace rpy::algebra;

namespace {

inline const TensorWord* cast_word(const BasisKeyCRef& key) noexcept
{
    return key.data<TensorWord>();
}

inline dimn_t cast_index(const BasisKeyCRef& key) noexcept
{
    return key.value<dimn_t>();
}

}// namespace

class TensorBasis::Details : containers::Vec<dimn_t>
{
    using base_t = containers::Vec<dimn_t>;
    deg_t m_depth;
    containers::Vec<dimn_t> m_powers;

    void grow(deg_t depth)
    {
        const auto& width = m_powers[1];
        for (; m_depth <= depth; ++m_depth) {
            emplace_back(1 + width * back());
            m_powers.emplace_back(m_powers.back() * width);
        }
    }

public:
    using typename base_t::const_iterator;
    using typename base_t::iterator;

    Details(deg_t width, deg_t depth) : m_depth{2}
    {
        reserve(depth + 2);
        insert(end(), {0, 1, static_cast<dimn_t>(1 + width)});

        m_powers.reserve(depth + 1);
        m_powers.insert(
                m_powers.end(),
                {static_cast<dimn_t>(1), static_cast<dimn_t>(width)}
        );
        grow(depth);
    }

    Details(const Details& old, deg_t depth) : m_depth(old.m_depth)
    {
        RPY_DBG_ASSERT(depth > old.m_depth);
        reserve(depth + 2);
        insert(end(), old.begin(), old.end());

        m_powers.reserve(depth + 1);
        insert(m_powers.end(), old.m_powers.begin(), old.m_powers.end());

        grow(depth);
    }

    using base_t::begin;
    using base_t::end;

    deg_t max_depth() const noexcept { return static_cast<deg_t>(size() - 2); }
    deg_t alphabet_size() const noexcept
    {
        return static_cast<deg_t>(m_powers[1]);
    }
    bool is_letter(dimn_t index) const noexcept
    {
        return 0 < index && index <= alphabet_size();
    }

    Slice<const dimn_t> sizes(deg_t depth) const noexcept
    {
        RPY_DBG_ASSERT(depth < size());
        return {data() + 1, static_cast<dimn_t>(2 + depth)};
    }

    Slice<const dimn_t> start_of_degrees(deg_t depth) const noexcept
    {
        return {data(), static_cast<dimn_t>(1 + depth)};
    }

    Slice<const dimn_t> powers() const noexcept
    {
        return {m_powers.data(), m_powers.size()};
    }

    static std::shared_ptr<const Details> get(deg_t width, deg_t depth)
    {
        static Mutex lock;
        static containers::HashMap<deg_t, std::shared_ptr<Details>> cache;

        LockGuard<Mutex> access(lock);

        auto entry = cache[width];
        if (entry) {
            if (entry->max_depth() < depth) {
                entry = std::make_shared<Details>(*entry, depth);
            }
        } else {
            entry = std::make_shared<Details>(width, depth);
        }

        return entry;
    }

    template <typename Predicate = std::less_equal<>>
    const_iterator
    boundary_before_index(dimn_t index, Predicate pred = Predicate{})
            const noexcept
    {
        auto it = ranges::lower_bound(begin() + 1, end(), index, pred);
        --it;
        RPY_DBG_ASSERT(pred(*it, index));
        return it;
    }

    dtl::ToLetterRange iterate_letters(dimn_t index) const noexcept
    {
        auto boundary = boundary_before_index(index);
        return {powers(),
                index - *boundary,
                alphabet_size(),
                static_cast<deg_t>(boundary - begin())};
    }
};

bool TensorBasis::is_word(const BasisKeyCRef& key) const noexcept
{
    return key.type() == m_supported_key_types[0];
}
bool TensorBasis::is_index(const BasisKeyCRef& key) const noexcept
{
    return key.type() == m_supported_key_types[1];
}

string TensorBasis::to_string_nofail(const BasisKeyCRef& key) const noexcept
{

    if (is_index(key)) { return std::to_string(cast_index(key)); }
    if (is_word(key)) {
        std::stringstream ss;
        cast_word(key)->print(ss);
        return ss.str();
    }

    const auto& tp = key.type();
    if (tp) { return string_cat("of type ", tp->name()); }

    return "undefined type";
}

TensorBasis::TensorBasis(deg_t width, deg_t depth)
    : Basis(basis_id, {true, true, true}),
      m_supported_key_types{TensorWordType::get(), IndexKeyType::get()},
      p_details(Details::get(width, depth)),
      m_width(width),
      m_depth(depth)
{}

BasisComparison TensorBasis::compare(BasisPointer other) const noexcept
{
    if (this == other) { return BasisComparison::IsSame; }
    if (other->id() == basis_id) {
        const auto* ptr = static_cast<const TensorBasis*>(other.get());
        if (ptr->m_width == m_width && ptr->m_depth == m_depth) {
            return BasisComparison::IsSame;
        }

        if (ptr->m_width == m_width) { return BasisComparison::IsSame; }
    }

    return BasisComparison::IsNotCompatible;
}
bool TensorBasis::supports_key_type(const devices::TypePtr& type) const noexcept
{
    return ranges::contains(m_supported_key_types, type);
}
Slice<const devices::TypePtr> TensorBasis::supported_key_types() const noexcept
{
    return {m_supported_key_types.data(), m_supported_key_types.size()};
}
bool TensorBasis::has_key(BasisKeyCRef key) const noexcept
{
    if (is_word(key)) {
        const auto* word = cast_word(key);

        if (word->degree() > m_depth) { return false; }

        return ranges::all_of(*word, [width = m_width](const auto& letter) {
            return 0 < letter && letter <= static_cast<decltype(letter)>(width);
        });
    }
    if (is_index(key)) { return cast_index(key) < max_dimension(); }

    return false;
}
string TensorBasis::to_string(BasisKeyCRef key) const
{
    if (is_word(key)) {
        std::stringstream ss;
        cast_word(key)->print(ss);
        return ss.str();
    }
    if (is_index(key)) {
        auto index = cast_index(key);

        std::stringstream ss;
        bool first = true;
        for (const auto letter : p_details->iterate_letters(index)) {
            if (!first) {
                ss << ',';
            } else {
                first = false;
            }
            ss << letter;
        }
        return ss.str();
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
bool TensorBasis::equals(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    return to_index(k1) == to_index(k2);
}
hash_t TensorBasis::hash(BasisKeyCRef k1) const
{
    return static_cast<hash_t>(to_index(k1));
}
dimn_t TensorBasis::max_dimension() const noexcept
{
    return p_details->sizes(m_depth)[m_depth];
}
dimn_t TensorBasis::dense_dimension(dimn_t size) const
{
    RPY_CHECK(size <= max_dimension());
    return *ranges::lower_bound(*p_details, size);
}
bool TensorBasis::less(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    return to_index(k1) < to_index(k2);
}
dimn_t TensorBasis::to_index(BasisKeyCRef key) const
{
    if (is_index(key)) { return cast_index(key); }
    if (is_word(key)) {
        const auto* word = cast_word(key);

        return ranges::fold_left(
                *word,
                static_cast<dimn_t>(0),
                [width = static_cast<dimn_t>(m_width
                 )](const auto& acc, const auto& letter) {
                    return acc * width + static_cast<dimn_t>(letter);
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
BasisKey TensorBasis::to_key(dimn_t index) const
{
    RPY_CHECK(index < max_dimension());
    const auto letter_range = p_details->iterate_letters(index);
    return BasisKey(TensorWord(letter_range.begin(), letter_range.end()));
}
KeyRange TensorBasis::iterate_keys() const { return Basis::iterate_keys(); }
algebra::dtl::BasisIterator TensorBasis::keys_begin() const
{
    return Basis::keys_begin();
}
algebra::dtl::BasisIterator TensorBasis::keys_end() const
{
    return Basis::keys_end();
}
deg_t TensorBasis::max_degree() const { return m_depth; }
deg_t TensorBasis::degree(BasisKeyCRef key) const
{
    if (is_word(key)) { return cast_word(key)->degree(); }
    if (is_index(key)) {
        const auto index = cast_index(key);
        RPY_CHECK(index < max_dimension());

        auto it = p_details->boundary_before_index(index);
        return static_cast<deg_t>(it - p_details->begin());
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
deg_t TensorBasis::dimension_to_degree(dimn_t dimension) const
{
    auto it = p_details->boundary_before_index(dimension);
    return static_cast<deg_t>(it - p_details->begin());
}
dimn_t TensorBasis::degree_to_dimension(deg_t degree) const
{
    RPY_CHECK(degree <= m_depth);
    const auto sizes = p_details->sizes(m_depth);
    return sizes[degree];
}
KeyRange TensorBasis::iterate_keys_of_degree(deg_t degree) const
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t TensorBasis::alphabet_size() const { return m_width; }
bool TensorBasis::is_letter(BasisKeyCRef key) const
{
    if (is_word(key)) { return cast_word(key)->is_letter(); }
    if (is_index(key)) {
        const auto index = cast_index(key);
        return 0 < index && index <= static_cast<dimn_t>(m_width);
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
let_t TensorBasis::get_letter(BasisKeyCRef key) const
{
    RPY_DBG_ASSERT(is_letter(key));
    if (is_word(key)) { return cast_word(key)->get_letter(); }
    if (is_index(key)) {
        const auto index = cast_index(key);
        return static_cast<let_t>(index);
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
pair<BasisKey, BasisKey> TensorBasis::parents(BasisKeyCRef key) const
{
    if (is_word(key)) {
        const auto* word = cast_word(key);
        if (word->is_letter()) { return {BasisKey(), BasisKey(*word)}; }
        return {BasisKey(word->left_parent()), BasisKey(word->right_parent())};
    }

    if (is_index(key)) {
        auto index = cast_index(key);
        if (index == 0) { return {BasisKey(key), BasisKey(key)}; }
        if (p_details->is_letter(index)) {
            return {BasisKey(), BasisKey(key.type(), index)};
        }

        auto it = p_details->boundary_before_index(index);

        auto degree = static_cast<deg_t>(it - p_details->begin()) - 1;

        index -= *it;

        auto split = p_details->powers()[degree - 1];

        auto [rem, quo] = remquo(index, split);

        return {BasisKey(index_key_type(), 1 + quo),
                BasisKey(index_key_type(), *(--it) + rem)};
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
BasisPointer TensorBasis::get(deg_t width, deg_t depth)
{
    return new TensorBasis(width, depth);
}
