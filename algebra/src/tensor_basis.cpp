//
// Created by sam on 16/02/24.
//


#include "tensor_basis.h"
#include <roughpy/core/macros.h>
#include <roughpy/core/helpers.h>

#include <roughpy/platform/errors.h>
#include "tensor_word.h"

#include <algorithm>
#include <functional>

using namespace rpy;
using namespace rpy::algebra;




namespace {


const TensorWord* cast_pointer_key(const BasisKey& key) {
    RPY_CHECK(key.is_valid_pointer());
    const auto* ptr = key.get_pointer();
    RPY_CHECK(ptr->key_type() == "tensor_word");
    return reinterpret_cast<const TensorWord*>(ptr);
}

void print_index(string& out, dimn_t width, dimn_t index) noexcept {
    if (index == 0) { return; }

    dimn_t tmp = index;
    index /= width;
    out += std::to_string(1 + tmp - index*width);

    while (index > width) {
        auto tmp = index;
        index /= width;
        out.push_back(',');
        out += std::to_string(1 + tmp - index*width);
    }

}

void print_word(string& out, const TensorWord* word) noexcept {
    if (word->degree() == 0) { return; }

    auto it = word->begin();
    auto end = word->end();

    out += std::to_string(*(it++));

    for( ; it != end; ++it) {
        out.push_back(',');
        out += std::to_string(*it);
    }
}

deg_t index_to_degree(const std::vector<dimn_t>& degree_sizes, dimn_t arg) noexcept {
    if (arg == 0) { return 0; }
    auto begin = degree_sizes.begin();
    auto end = degree_sizes.end();

    auto it = std::lower_bound(begin, end, arg, std::less_equal<>());

    RPY_DBG_ASSERT(it != degree_sizes.end());
    return static_cast<deg_t>(it - begin);
}


}


bool rpy::algebra::TensorBasis::has_key(BasisKey key) const noexcept
{
    return false;
}
string rpy::algebra::TensorBasis::to_string(BasisKey key) const noexcept
{
    string result;
    if (key.is_index()) {
        print_index(result, m_width, key.get_index());
    } else {
        print_word(result, cast_pointer_key(key));
    }
    return result;
}
bool rpy::algebra::TensorBasis::equals(BasisKey k1, BasisKey k2) const noexcept
{
    return false;
}
hash_t rpy::algebra::TensorBasis::hash(BasisKey k1) const noexcept {
    return static_cast<hash_t>(to_index(k1));
}
bool TensorBasis::less(BasisKey k1, BasisKey k2) const noexcept
{
    return Basis::less(k1, k2);
}
dimn_t TensorBasis::to_index(BasisKey key) const
{
    if (key.is_index()) {
        auto index = key.get_index();
        RPY_DBG_ASSERT(index. <= max_dimension());
        return index;
    }

    const auto* word = cast_pointer_key(key);

    dimn_t index = 0;

    for (const auto& let : *word) {
        index *= m_width;
        index += let;
    }

    return index;
}
BasisKey TensorBasis::to_key(dimn_t index) const
{
    RPY_CHECK(index < m_max_dimension);
    auto degree = index_to_degree(m_degree_sizes, index);
    BasisKey key(new TensorWord(degree));


    index -= m_degree_sizes[degree];
    dimn_t tmp;
    while (index > m_width) {
        tmp = index;
        index /= m_width;

    }

}
KeyRange TensorBasis::iterate_keys() const noexcept
{
    return Basis::iterate_keys();
}
deg_t TensorBasis::max_degree() const noexcept { return Basis::max_degree(); }
deg_t TensorBasis::degree(BasisKey key) const noexcept
{
    return Basis::degree(key);
}
KeyRange TensorBasis::iterate_keys_of_degree(deg_t degree) const noexcept
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t TensorBasis::alphabet_size() const noexcept
{
    return m_width;
}
bool TensorBasis::is_letter(BasisKey key) const noexcept
{
    return Basis::is_letter(key);
}
let_t TensorBasis::get_letter(BasisKey key) const noexcept
{
    return Basis::get_letter(key);
}
pair<optional<BasisKey>, optional<BasisKey>> TensorBasis::parents(BasisKey key
) const noexcept
{
    return Basis::parents(key);
}
