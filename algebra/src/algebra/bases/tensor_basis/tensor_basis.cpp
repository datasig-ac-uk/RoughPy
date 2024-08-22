//
// Created by sam on 16/02/24.
//

#include "tensor_basis.h"
#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/container/vector.h>
#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>

#include <roughpy/platform/errors.h>

#include <algorithm>
#include <functional>
#include <mutex>

using namespace rpy;
using namespace rpy::algebra;

TensorBasis::TensorBasis(rpy::deg_t width, rpy::deg_t depth)
    : Basis(basis_id, {true, true, true}),
      m_width(width),
      m_depth(depth)
{
    m_degree_sizes.reserve(depth + 1);

    m_max_dimension = 1;
    for (deg_t i = 0; i <= depth; ++i) {
        m_degree_sizes.push_back(m_max_dimension);
        m_max_dimension = 1 + width * m_max_dimension;
    }
}

dimn_t TensorBasis::max_dimension() const noexcept { return m_max_dimension; }

dimn_t TensorBasis::dense_dimension(dimn_t size) const
{
    const auto end = m_degree_sizes.end();
    auto pos = ranges::lower_bound(m_degree_sizes.begin(), end, size);
    RPY_CHECK(pos != end);
    return *pos;
}

namespace {

template <typename F>
void do_for_each_letter_in_index(deg_t width, dimn_t index, F&& op)
{
    const auto divisor = static_cast<dimn_t>(width);

    while (index > 0) {
        auto [div, rem] = remquo(index - 1, divisor);
        op(rem + 1);
        index = div;
    }
}

deg_t index_to_degree(
        const containers::Vec<dimn_t>& degree_sizes,
        dimn_t arg
) noexcept
{
    if (arg == 0) { return 0; }
    auto begin = degree_sizes.begin();
    auto end = degree_sizes.end();

    auto it = std::lower_bound(begin, end, arg, std::less_equal<>());

    RPY_DBG_ASSERT(it != degree_sizes.end());
    return static_cast<deg_t>(it - begin);
}

const TensorWord* cast_pointer_key(const BasisKey& key)
{
    RPY_CHECK(key.is_valid_pointer());
    const auto* ptr = key.get_pointer();
    RPY_CHECK(ptr->key_type() == TensorWord::key_name);
    return reinterpret_cast<const TensorWord*>(ptr);
}
void print_word(string& out, const TensorWord* word) noexcept
{
    if (word->degree() == 0) { return; }

    auto it = word->begin();
    auto end = word->end();

    out += std::to_string(*(it++));

    for (; it != end; ++it) {
        out.push_back(',');
        out += std::to_string(*it);
    }
}

bool word_is_valid(deg_t width, deg_t depth, const TensorWord* word) noexcept
{
    if (word->degree() > depth) { return false; }

    return std::all_of(word->cbegin(), word->cend(), [width](auto letter) {
        return letter > 0 && static_cast<deg_t>(letter) <= width;
    });
}

}// namespace

bool rpy::algebra::TensorBasis::has_key(BasisKey key) const noexcept
{
    if (key.is_index()) { return key.get_index() < max_dimension(); }

    if (!key.is_valid_pointer()) { return false; }

    const auto* ptr = key.get_pointer();
    if (ptr->key_type() != "tensor_word") { return false; }

    return word_is_valid(m_width, m_depth, static_cast<const TensorWord*>(ptr));
}
string rpy::algebra::TensorBasis::to_string(BasisKey key) const
{
    string result;
    if (key.is_index()) {
        bool first = true;
        do_for_each_letter_in_index(
                m_width,
                key.get_index(),
                [&result, &first](dimn_t letter) {
                    if (first) {
                        first = false;
                    } else {
                        result.push_back(',');
                    }
                    result += std::to_string(letter);
                }
        );
    } else {
        print_word(result, cast_pointer_key(key));
    }
    return result;
}
bool rpy::algebra::TensorBasis::equals(BasisKey k1, BasisKey k2) const
{
    return to_index(k1) == to_index(k2);
}
hash_t rpy::algebra::TensorBasis::hash(BasisKey k1) const
{
    return static_cast<hash_t>(to_index(k1));
}
bool TensorBasis::less(BasisKey k1, BasisKey k2) const
{
    return to_index(k1) < to_index(k2);
}
dimn_t TensorBasis::to_index(BasisKey key) const
{
    if (key.is_index()) {
        auto index = key.get_index();
        RPY_CHECK(index <= max_dimension());
        return index;
    }

    const auto* word = cast_pointer_key(key);

    dimn_t index = 0;
    for (const auto& let : *word) {
        RPY_CHECK(let <= m_width);
        index *= m_width;
        index += let;
    }

    return index;
}
BasisKey TensorBasis::to_key(dimn_t index) const
{
    RPY_CHECK(index < m_max_dimension);
    auto degree = index_to_degree(m_degree_sizes, index);
    auto word = std::make_unique<TensorWord>(degree);

    index -= m_degree_sizes[degree];
    dimn_t tmp;
    while (index > m_width) {
        tmp = index;
        index /= m_width;
        word->push_back(1 + tmp - index * m_width);
    }

    std::reverse(word->begin(), word->end());

    return BasisKey(word.release());
}
KeyRange TensorBasis::iterate_keys() const { return Basis::iterate_keys(); }
deg_t TensorBasis::max_degree() const noexcept { return m_depth; }
deg_t TensorBasis::degree(BasisKey key) const
{
    RPY_CHECK(has_key(key));
    if (key.is_pointer()) {
        RPY_CHECK(key.is_valid_pointer());

        const auto* ptr = cast_pointer_key(key);
        return static_cast<deg_t>(ptr->degree());
    }

    return index_to_degree(m_degree_sizes, key.get_index());
}

KeyRange TensorBasis::iterate_keys_of_degree(deg_t degree) const
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t TensorBasis::alphabet_size() const noexcept { return m_width; }
bool TensorBasis::is_letter(BasisKey key) const { return degree(key) == 1; }
let_t TensorBasis::get_letter(BasisKey key) const
{
    RPY_DBG_ASSERT(is_letter(key));

    return 0;
}
pair<optional<BasisKey>, optional<BasisKey>> TensorBasis::parents(BasisKey key
) const
{
    return Basis::parents(key);
}

static std::mutex s_tensor_basis_lock;
static containers::HashMap<pair<deg_t, deg_t>, BasisPointer>
        s_tensor_basis_cache;

BasisPointer TensorBasis::get(deg_t width, deg_t depth)
{
    std::lock_guard<std::mutex> access(s_tensor_basis_lock);
    auto& basis = s_tensor_basis_cache[{width, depth}];
    if (!basis) { basis = new TensorBasis(width, depth); }
    return basis;
}

BasisComparison TensorBasis::compare(BasisPointer other) const noexcept
{
    if (other == this) { return BasisComparison::IsSame; }

    if (other->id() == basis_id && other->alphabet_size() == m_width) {
        return BasisComparison::IsCompatible;
    }

    return BasisComparison::IsNotCompatible;
}
