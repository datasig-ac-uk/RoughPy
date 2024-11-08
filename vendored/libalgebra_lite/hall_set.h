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

#ifndef LIBALGEBRA_LITE_HALL_SET_H
#define LIBALGEBRA_LITE_HALL_SET_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/flat_map.hpp>

#include "index_key.h"
#include "registry.h"
#include "basis_traits.h"

namespace lal {



class LIBALGEBRA_LITE_EXPORT hall_set {
public:
    using letter_type = let_t;
    using degree_type = deg_t;
    using key_type = index_key<>;
    using size_type = dimn_t;
    using parent_type = std::pair<key_type, key_type>;
private:
    using data_type = std::vector<parent_type>;
    using reverse_map_type = boost::container::flat_map<parent_type, key_type>;
//    using reverse_map_type = std::map<parent_type, key_type>;
    using l2k_map_type = std::vector<key_type>;
    using size_vector_type = std::vector<size_type>;
    using degree_range_map_type = std::vector<std::pair<size_type, size_type>>;

    degree_type current_degree;

    std::vector<letter_type> letters;
    data_type data;
    reverse_map_type reverse_map;
    l2k_map_type l2k;
    degree_range_map_type degree_ranges;
    size_vector_type m_sizes;

public:

    struct find_result
    {
        typename reverse_map_type::const_iterator it;
        bool found;
    };


    static constexpr key_type root_element {0, 0};
    static constexpr parent_type root_parent {root_element, root_element};

    explicit hall_set(degree_type width, degree_type depth=1);
    explicit hall_set(const hall_set& existing, degree_type deg);

    void grow_up(degree_type deg);


    key_type key_of_letter(let_t) const noexcept;
    size_type size(deg_t) const noexcept;
    const std::vector<dimn_t>& sizes() const noexcept { return m_sizes; }

    size_type size_of_degree(deg_t) const noexcept;
    bool letter(const key_type&) const noexcept;
    letter_type get_letter(dimn_t idx) const noexcept;

    find_result find(parent_type parent) const noexcept;

    size_type index_of_key(key_type arg) const noexcept;
    key_type key_of_index(size_type index) const noexcept;

    const parent_type &operator[](const key_type&) const noexcept;
    const key_type& operator[](const parent_type&) const;
};


template<typename Func,
        typename Binop,
        typename ReturnType=decltype(std::declval<Func>()(std::declval<let_t>()))>
class hall_extension {
public:
    using key_type = typename hall_set::key_type;
    using return_type = ReturnType;
private:
    using cached_type = decltype(std::declval<Func>()(std::declval<let_t>()));

    std::shared_ptr<const hall_set> m_hall_set;
    Func m_func;
    Binop m_binop;
    mutable std::unordered_map<key_type, cached_type> m_cache;
    mutable std::recursive_mutex m_lock;
public:

    explicit hall_extension(std::shared_ptr<const hall_set> hs, Func&& func, Binop&& binop);

    return_type operator()(key_type key) const;

};


class LIBALGEBRA_LITE_EXPORT hall_basis
{
    deg_t m_width;
    deg_t m_depth;
    std::shared_ptr<const hall_set> p_hallset;

    static std::string letter_to_string(let_t letter);
    static std::string key_to_string_op(const std::string& left, const std::string& right);


    hall_extension<decltype(&letter_to_string),
            decltype(&key_to_string_op),
            const std::string&> m_key_to_string;

public:

    using key_type = typename hall_set::key_type;
    using parent_type = typename hall_set::parent_type;
    using degree_tag LAL_UNUSED = with_degree_tag;

    hall_basis(deg_t width, deg_t depth) : m_width(width), m_depth(depth),
        p_hallset(new hall_set(width, depth)),
        m_key_to_string(p_hallset, &letter_to_string, &key_to_string_op)
    {}

    deg_t width() const noexcept { return m_width; }
    deg_t depth() const noexcept { return m_depth; }

    static constexpr deg_t degree(const key_type& arg) noexcept
    { return deg_t(arg.degree()); }

    bool letter(const key_type& arg) const noexcept {
        return arg.degree() == 1;
    }
    parent_type parents(const key_type& arg) const noexcept
    { return (*p_hallset)[arg]; }
    key_type lparent(const key_type& arg) const noexcept
    { return parents(arg).first; }
    key_type rparent(const key_type& arg) const noexcept
    { return parents(arg).second; }
    key_type key_of_letter(let_t letter) const noexcept
    { return p_hallset->key_of_letter(letter); }
    let_t first_letter(const key_type& key) const noexcept
    { return p_hallset->get_letter((*p_hallset)[key].first.index()); }
    let_t to_letter(const key_type& key) const noexcept
    { return p_hallset->get_letter(key.index()); }
    dimn_t size(int deg) const noexcept
    {
        return p_hallset->size(deg < 0 ? m_depth : static_cast<deg_t>(deg));
    }

    const std::vector<dimn_t>& sizes() const noexcept
    { return p_hallset->sizes(); }

    dimn_t start_of_degree(deg_t deg) const noexcept
    {
        return (deg == 0) ? 0 : p_hallset->size(deg-1);
    }
    dimn_t size_of_degree(deg_t deg) const noexcept { return p_hallset->size_of_degree(deg); }
    typename hall_set::find_result find(parent_type parents) const noexcept;

    dimn_t key_to_index(key_type arg) const noexcept { return p_hallset->index_of_key(arg); }
    key_type index_to_key(dimn_t arg) const noexcept { return p_hallset->key_of_index(arg); }


    std::shared_ptr<const hall_set> get_hall_set() const noexcept { return p_hallset; }

    std::ostream& print_key(std::ostream& os, key_type key) const;


    void advance_key(key_type& key) const;
};






template<typename Func, typename Binop, typename ReturnType>
hall_extension<Func, Binop, ReturnType>::hall_extension(std::shared_ptr<const hall_set> hs, Func&& func, Binop&& binop)
        : m_hall_set(std::move(hs)),
        m_func(std::forward<Func>(func)),
        m_binop(std::forward<Binop>(binop))
{
}

template<typename Func, typename Binop, typename ReturnType>
ReturnType
hall_extension<Func, Binop, ReturnType>::operator()(
        hall_extension::key_type key) const
{
    std::lock_guard<std::recursive_mutex> access(m_lock);

    assert(key.degree() != 0);
    auto found = m_cache.find(key);
    if (found!=m_cache.end()) {
        return found->second;
    }

    if (m_hall_set->letter(key)) {
        return m_cache[key] = m_func(m_hall_set->get_letter(key.index()));
    }

    auto parents = (*m_hall_set)[key];
    return m_cache[key] = m_binop(operator()(parents.first), operator()(parents.second));
}

LAL_EXPORT_TEMPLATE_CLASS(basis_registry, hall_basis)

} // namespace lal

#endif //LIBALGEBRA_LITE_HALL_SET_H
