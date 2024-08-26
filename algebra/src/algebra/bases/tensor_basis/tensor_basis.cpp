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

    void grow(deg_t depth)
    {
        const auto& width = operator[](1);
        for (; m_depth <= depth; ++m_depth) {
            emplace_back(1 + width * back());
        }
    }

public:
    Details(deg_t width, deg_t depth) : m_depth{2}
    {
        reserve(depth + 1);
        emplace_back(1);
        emplace_back(width);
        grow(depth);
    }

    Details(const Details& old, deg_t depth) : m_depth(old.m_depth)
    {
        RPY_DBG_ASSERT(depth > old.m_depth);
        reserve(depth + 1);
        insert(end(), old.begin(), old.end());

        grow(depth);
    }

    deg_t max_depth() const noexcept { return static_cast<deg_t>(size() - 1); }

    Slice<const dimn_t> sizes(deg_t depth) const noexcept
    {
        RPY_DBG_ASSERT(depth < size());
        return {data(), static_cast<dimn_t>(1 + depth)};
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
      m_width(width),
      m_depth(depth),
      m_supported_key_types{TensorWordType::get(), IndexKeyType::get()},
      p_details(Details::get(width, depth))
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
    if (is_word(key)) { return true; }
    if (is_index(key)) { return true; }

    return false;
}
string TensorBasis::to_string(BasisKeyCRef key) const
{
    if (is_word(key)) {
        std::stringstream ss;
        cast_word(key)->print(ss);
        return ss.str();
    }
    if (is_index(key)) { const auto index = cast_index(key); }
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
    return Basis::max_dimension();
}
dimn_t TensorBasis::dense_dimension(dimn_t size) const
{
    return Basis::dense_dimension(size);
}
bool TensorBasis::less(BasisKeyCRef k1, BasisKeyCRef k2) const
{
    return to_index(k1) < to_index(k2);
}
dimn_t TensorBasis::to_index(BasisKeyCRef key) const
{
    return Basis::to_index(key);
}
BasisKey TensorBasis::to_key(dimn_t index) const
{
    return Basis::to_key(index);
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

        const auto sizes = p_details->sizes(m_depth);
        const auto begin = sizes.begin();
        const auto end = sizes.end();

        auto it = ranges::lower_bound(begin, end, index);
        return static_cast<deg_t>(it - begin);
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
    const auto sizes = p_details->sizes(m_depth);
    const auto begin = sizes.begin();
    const auto end = sizes.end();

    const auto it = ranges::lower_bound(begin, end, dimension, std::less_equal{});

    return static_cast<deg_t>(it - begin);
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
    if (is_word(key)) {
        return cast_word(key)->get_letter();
    }
    if (is_index(key)) {
        const auto index = cast_index(key);
        return static_cast<let_t>(index);
    }
}
pair<BasisKey, BasisKey> TensorBasis::parents(BasisKeyCRef key) const
{
    return Basis::parents(key);
}
BasisPointer TensorBasis::get(deg_t width, deg_t depth)
{
    return new TensorBasis(width, depth);
}
