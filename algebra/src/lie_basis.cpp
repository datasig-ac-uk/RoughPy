//
// Created by sam on 3/13/24.
//

#include "lie_basis.h"

using namespace rpy;
using namespace rpy::algebra;

bool LieBasis::has_key(BasisKey key) const noexcept { return false; }
string LieBasis::to_string(BasisKey key) const noexcept
{
    return std::string();
}
bool LieBasis::equals(BasisKey k1, BasisKey k2) const noexcept
{
    return to_index(k1) == to_index(k2);
}
hash_t LieBasis::hash(BasisKey k1) const noexcept
{
    return static_cast<hash_t>(to_index(k1));
}
bool LieBasis::less(BasisKey k1, BasisKey k2) const noexcept
{
    return to_index(k1) < to_index(k2);
}
dimn_t LieBasis::to_index(BasisKey key) const { return Basis::to_index(key); }
BasisKey LieBasis::to_key(dimn_t index) const { return Basis::to_key(index); }
KeyRange LieBasis::iterate_keys() const noexcept
{
    return Basis::iterate_keys();
}
deg_t LieBasis::max_degree() const noexcept { return m_depth; }
deg_t LieBasis::degree(BasisKey key) const noexcept
{
    return Basis::degree(key);
}
KeyRange LieBasis::iterate_keys_of_degree(deg_t degree) const noexcept
{
    return Basis::iterate_keys_of_degree(degree);
}
deg_t LieBasis::alphabet_size() const noexcept { return m_width; }
bool LieBasis::is_letter(BasisKey key) const noexcept
{
    return Basis::is_letter(key);
}
let_t LieBasis::get_letter(BasisKey key) const noexcept
{
    return Basis::get_letter(key);
}
pair<optional<BasisKey>, optional<BasisKey>> LieBasis::parents(BasisKey key
) const noexcept
{
    return Basis::parents(key);
}
