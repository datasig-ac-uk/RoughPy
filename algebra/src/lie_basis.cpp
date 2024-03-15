//
// Created by sam on 3/13/24.
//

#include "lie_basis.h"

#include "hall_set_size.h"

#include <boost/container/flat_map.hpp>

using namespace rpy;
using namespace rpy::algebra;

class LieBasis::HallSet
{
    mutable std::recursive_mutex m_lock;
    boost::container::flat_map<parent_type, BasisKey> m_reverse_map;
    std::vector<let_t> m_letters;
    std::vector<dimn_t> m_degree_sizes;
    std::vector<parent_type> m_hall_set;
    std::vector<BasisKey> m_l2k;
    std::vector<pair<dimn_t, dimn_t>> m_degree_ranges;

    deg_t m_degree = 0;
    deg_t m_width;
    HallSetSizeHelper size_helper;

public:
    explicit HallSet(deg_t width);

    void grow(deg_t degree);

    std::lock_guard<std::recursive_mutex> lock() const noexcept
    {
        return std::lock_guard<std::recursive_mutex>(m_lock);
    }

    RPY_NO_DISCARD dimn_t size(deg_t deg) const noexcept
    {
        // size_helper is thread safe and never changed
        return size_helper(deg);
    }

    RPY_NO_DISCARD BasisKey letter_to_key(let_t letter) const
    {
        const auto access = lock();
        RPY_CHECK(0 < letter && letter <= m_width);
        return m_l2k[letter - 1];
    }

    RPY_NO_DISCARD dimn_t size_of_degree(deg_t degree) const
    {
        if (degree == 0) { return 0; }

        const auto access = lock();
        if (degree < m_degree_ranges.size()) {
            const auto& range = m_degree_ranges[degree];
            return range.second - range.first;
        }

        return size_helper(degree) - size_helper(degree - 1);
    }

    RPY_NO_DISCARD optional<BasisKey>
    find_child(BasisKey left, BasisKey right) const
    {
        RPY_DBG_ASSERT(left.is_index() && right.is_index());
        const auto access = lock();
        optional<BasisKey> result;

        const auto it = m_reverse_map.find({left, right});
        if (it != m_reverse_map.end()) { return it->second; }

        return result;
    }

    RPY_NO_DISCARD optional<parent_type>
    find_parents(BasisKey key, deg_t max_degree) const
    {
        RPY_DBG_ASSERT(key.is_index());

        const auto access = lock();
        const auto index = key.get_index();

        if (index < m_degree_sizes[std::min(max_degree, m_degree)]) {
            return m_hall_set[index];
        }
        return {};
    }
};

LieBasis::LieBasis(deg_t width, deg_t depth) : m_width(width), m_depth(depth)
{
    static std::mutex s_lock;
    static std::unordered_map<deg_t, std::shared_ptr<HallSet>> s_cache;

    std::lock_guard<std::mutex> access(s_lock);
    auto& hallset = s_cache[width];
    if (!hallset) { hallset = std::make_shared<HallSet>(width); }
    hallset->grow(depth);
    p_hallset = hallset;
}

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
dimn_t LieBasis::to_index(BasisKey key) const
{
    if (key.is_index()) { return key.get_index(); }

    return 0;
}
BasisKey LieBasis::to_key(dimn_t index) const { return BasisKey(index); }
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

LieBasis::HallSet::HallSet(deg_t width)
    : m_width(width),
      size_helper(width),
      m_degree_sizes{
              0,
              static_cast<dimn_t>(width)
},
      m_degree_ranges{{0, 0}}
{
    m_hall_set.reserve(width);
    m_letters.reserve(width);
    m_l2k.reserve(width);

    for (let_t l = 1; l <= static_cast<let_t>(width); ++l) {
        BasisKey key(l - 1);
        parent_type parents{0, l};
        m_letters.push_back(l);
        m_hall_set.push_back(parents);
        m_reverse_map.insert(std::make_pair(parents, l));
        m_l2k.push_back(key);
    }

    m_degree_ranges.emplace_back(1, m_hall_set.size());
    ++m_degree;
}

void LieBasis::HallSet::grow(rpy::deg_t degree)
{
    const auto access = lock();
    if (degree <= m_degree) { return; }

    for (auto d = m_degree + 1; d <= degree; ++d) {
        dimn_t k_index = m_hall_set.size();

        for (deg_t e = 1; 2 * e <= d; ++e) {
            auto i_bounds = m_degree_ranges[e];
            auto j_bounds = m_degree_ranges[d - e];

            for (auto i = i_bounds.first; i < i_bounds.second; ++i) {
                BasisKey ikey(i);
                for (auto j = std::max(j_bounds.first, i + 1);
                     j < j_bounds.second;
                     ++j) {
                    BasisKey jkey(j);
                    if (m_hall_set[j].first.get_index() <= ikey.get_index()) {
                        BasisKey key(k_index++);
                        parent_type parents(ikey, jkey);
                        m_hall_set.push_back(parents);
                        m_reverse_map.insert(std::make_pair(parents, key));
                    }
                }
            }

            m_degree_ranges.emplace_back(
                    m_degree_ranges.back().second,
                    k_index
            );
            m_degree_sizes.push_back(k_index);
            ++m_degree;
        }
    }
}
