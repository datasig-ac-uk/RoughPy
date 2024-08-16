//
// Created by sam on 8/15/24.
//

#include "hall_basis.h"

#include "lie_word.h"
#include "lie_word_type.h"

using namespace rpy;
using namespace rpy::algebra;

HallBasis::HallBasis(deg_t width, deg_t depth)
    : Basis("hall_basis", {true, true, true}),
      m_width(width),
      m_max_degree(depth),
      p_lie_word_type(LieWordType::get()),
      p_index_key_type(nullptr)
{}

bool HallBasis::has_key(BasisKey key) const noexcept {}
string HallBasis::to_string(BasisKey key) const {}
bool HallBasis::equals(BasisKey k1, BasisKey k2) const {}
hash_t HallBasis::hash(BasisKey k1) const {}
dimn_t HallBasis::max_dimension() const noexcept
{
    return Basis::max_dimension();
}
dimn_t HallBasis::dense_dimension(dimn_t size) const
{
    return Basis::dense_dimension(size);
}
bool HallBasis::less(BasisKey k1, BasisKey k2) const
{
    return Basis::less(k1, k2);
}
dimn_t HallBasis::to_index(BasisKey key) const { return Basis::to_index(key); }
BasisKey HallBasis::to_key(dimn_t index) const { return Basis::to_key(index); }
KeyRange HallBasis::iterate_keys() const { return Basis::iterate_keys(); }
algebra::dtl::BasisIterator HallBasis::keys_begin() const
{
    return Basis::keys_begin();
}
algebra::dtl::BasisIterator HallBasis::keys_end() const
{
    return Basis::keys_end();
}
deg_t HallBasis::max_degree() const { return Basis::max_degree(); }
deg_t HallBasis::degree(BasisKey key) const { return Basis::degree(key); }
dimn_t HallBasis::dimension_to_degree(deg_t degree) const
{
    return Basis::dimension_to_degree(degree);
}
KeyRange HallBasis::iterate_keys_of_degree(deg_t degree) const
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t HallBasis::alphabet_size() const { return Basis::alphabet_size(); }
bool HallBasis::is_letter(BasisKey key) const { return Basis::is_letter(key); }
let_t HallBasis::get_letter(BasisKey key) const
{
    return Basis::get_letter(key);
}
pair<optional<BasisKey>, optional<BasisKey>> HallBasis::parents(BasisKey key
) const
{
    return Basis::parents(key);
}
BasisComparison HallBasis::compare(BasisPointer other) const noexcept
{
    return Basis::compare(other);
}
Rc<VectorContext> HallBasis::default_vector_context() const
{
    return Basis::default_vector_context();
}
