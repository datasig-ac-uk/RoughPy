// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 26/07/22.
//

#include "libalgebra_lite/hall_set.h"

#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <functional>

namespace lal {


hall_set::hall_set(hall_set::degree_type width, hall_set::degree_type depth)
        :  current_degree(0)
{
    data.reserve(1 + width);
    letters.reserve(width);
    m_sizes.reserve(2);
    l2k.reserve(width);
    degree_ranges.reserve(2);

    key_type zero_key(0, 0);

    data.push_back({zero_key, zero_key});
    degree_ranges.push_back({0, 1});
    m_sizes.push_back(0);

    for (letter_type l = 1; l <= static_cast<letter_type>(width); ++l) {
        key_type key(1, l-1);
//        std::cout << zero_key << ' ' << key << ' ' << key << '\n';
        parent_type parents{zero_key, key};
        letters.push_back(l);
        data.push_back(parents);
        reverse_map.insert(std::make_pair(parents, key));
        l2k.push_back(key);
    }

    std::pair<size_type, size_type> range {
            degree_ranges[current_degree].second,
            data.size()
    };

    degree_ranges.push_back(range);
    m_sizes.push_back(width);
    ++current_degree;

    if (depth > 1) {
        grow_up(depth);
    }
}

void hall_set::grow_up(hall_set::degree_type new_depth)
{

    for (degree_type d = current_degree + 1; d <= new_depth; ++d) {
        size_type k_index = 0;
        for (degree_type e = 1; 2 * e <= d; ++e) {
            letter_type i_lower, i_upper, j_lower, j_upper;
            i_lower = degree_ranges[e].first;
            i_upper = degree_ranges[e].second;
            j_lower = degree_ranges[d - e].first;
            j_upper = degree_ranges[d - e].second;

            for (letter_type i = i_lower; i < i_upper; ++i) {
                key_type ik(e, i - i_lower);
                for (letter_type j = std::max(j_lower, i + 1); j < j_upper; ++j) {
                    key_type jk(d-e, j-j_lower);
                    if (data[j].first <= ik) {
                        key_type new_key (d, k_index++);
                        parent_type parents(ik, jk);
                        data.push_back(parents);
                        reverse_map.insert(std::make_pair(parents, new_key));
//                        std::cout << parents.first << ' ' << parents.second  << ' ' << new_key << '\n';
                    }
                }
            }
        }

        std::pair<size_type, size_type> range;
        range.first = degree_ranges[current_degree].second;
        range.second = data.size();
        degree_ranges.push_back(range);
        // The hall set contains an entry for the "god element" 0,
        // so subtract one from the size.
        m_sizes.push_back(data.size() - 1);

        ++current_degree;
    }
}

hall_set::key_type hall_set::key_of_letter(let_t let) const noexcept
{
    return typename hall_set::key_type {1, let-1};
}
hall_set::size_type hall_set::size(deg_t deg) const noexcept
{
    if (deg >= 0 && static_cast<dimn_t>(deg) < m_sizes.size()) {
        return m_sizes[deg];
    }
    if (deg < 0 && deg >= -static_cast<idimn_t>(m_sizes.size())) {
        return m_sizes[m_sizes.size() + deg];
    }
    return m_sizes.back();
}
hall_set::hall_set(const hall_set& existing, hall_set::degree_type deg)
{

}
hall_set::size_type hall_set::size_of_degree(deg_t arg) const noexcept
{
    auto range = degree_ranges[arg];
    return range.second - range.first;
}
hall_set::letter_type hall_set::get_letter(dimn_t idx) const noexcept
{
    return let_t(idx+1);
}
typename hall_set::find_result hall_set::find(hall_set::parent_type parent) const noexcept
{
    find_result result;
    result.it = reverse_map.find(parent);
    result.found = (result.it != reverse_map.end());
    return result;
}
bool hall_set::letter(const hall_set::key_type &key) const noexcept
{
    return key.degree() == 1;
}
const hall_set::parent_type &hall_set::operator[](const hall_set::key_type &key) const noexcept
{
    const auto degree = key.degree();
    auto index = key.index();
    auto offset = degree_ranges[degree].first;
    assert(index + offset < degree_ranges[degree].second);
    assert(degree == data[index+offset].first.degree() + data[index+offset].second.degree());
    return data[index + offset];
//    return data[key.index() + size(deg_t(key.degree()-1)) + 1];
}
const hall_set::key_type &hall_set::operator[](const hall_set::parent_type &parent) const
{
    auto found = reverse_map.find(parent);
    if (found != reverse_map.end()) {
        assert(found->second.degree() == parent.first.degree() +parent.second.degree());
        return found->second;
    }
    return root_element;
}
dimn_t hall_set::index_of_key(hall_set::key_type arg) const noexcept
{
    assert(arg.degree() > 0);
    return arg.index() + size(deg_t(arg.degree())-1);
}
hall_set::key_type hall_set::key_of_index(hall_set::size_type index) const noexcept
{
    auto found = std::lower_bound(
            ++m_sizes.begin(),
            m_sizes.end(),
            index,
            std::less_equal<>()
            );
    if (found == m_sizes.end()) {
        return root_element;
    }
    assert(found != m_sizes.begin());
    auto deg = static_cast<deg_t>(found - m_sizes.begin());
    auto range_begin = *(--found);
    return key_type(deg, index - range_begin);
}

std::string hall_basis::letter_to_string(let_t letter)
{
    return std::to_string(letter);
}
std::string hall_basis::key_to_string_op(const std::string& left, const std::string& right)
{
    return "[" + left + "," + right + "]";
}

constexpr typename hall_set::key_type hall_set::root_element;
constexpr typename hall_set::parent_type hall_set::root_parent;

typename hall_set::find_result hall_basis::find(hall_basis::parent_type parents) const noexcept
{
    return p_hallset->find(parents);
}
std::ostream& hall_basis::print_key(std::ostream& os, hall_basis::key_type key) const
{
    return os << m_key_to_string(key);
}
template class basis_registry<hall_basis>;

void hall_basis::advance_key(hall_basis::key_type &key) const {
    const auto degree = static_cast<deg_t>(key.degree());
    const auto bound = p_hallset->size_of_degree(degree);
    ++key;
    if (key.index() >= bound) {
        key = key_type(degree+1, 0);
    }
}

} // namespace lal
