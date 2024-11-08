//
// Created by user on 25/07/22.
//

#ifndef LIBALGEBRA_LITE_BASIS_TRAITS_H
#define LIBALGEBRA_LITE_BASIS_TRAITS_H

#include "implementation_types.h"

#include <boost/type_traits/is_detected.hpp>

#include <algorithm>
#include <functional>
#include <utility>
#include <memory>

namespace lal {

struct with_degree_tag {};
struct without_degree_tag {};

namespace dtl {

template <typename Key, typename KeyOrder>
struct key_value_ordering
{
    template <typename S>
    using pair = std::pair<Key, S>;

    KeyOrder order;

    template <typename S>
    bool operator()(const pair<S>& lhs, const pair<S>& rhs) const noexcept
    {
        return order(lhs.first, rhs.first);
    }

};


template <typename Basis>
class has_degree_tag_helper {

    template <typename B=Basis>
    static typename B::degree_tag choose(void*);

    static without_degree_tag choose(...);

public:
    using type = decltype(choose(nullptr));
};

template <typename B>
using advance_key_t = decltype(std::declval<const B&>().advance_key(std::declval<typename B::key_type&>()));

template <typename Basis, bool=boost::is_detected<advance_key_t, Basis>::value>
struct advance_key_helper {

    static void advance_key(const Basis& b, typename Basis::key_type& key) { ++key; }
};

template <typename Basis>
struct advance_key_helper<Basis, true> {

    static void advance_key(const Basis& b, typename Basis::key_type& key) {
        b.advance_key(key);
    }

};



} // namespace dtl

template <typename Basis>
struct basis_trait {
    using key_type = typename Basis::key_type;
    using degree_tag = typename dtl::has_degree_tag_helper<Basis>::type;

    static dimn_t max_dimension(const Basis& basis) noexcept { return basis.size(-1); };

    static dimn_t key_to_index(const Basis& basis, const key_type& k) noexcept { return basis.key_to_index(k); }
    static key_type index_to_key(const Basis& basis, dimn_t idx) noexcept { return basis.index_to_key(idx); }
    static deg_t degree(const Basis& basis, const key_type& key) noexcept
    { return basis.degree(key); }
    static deg_t max_degree(const Basis& basis) noexcept
    { return basis.depth(); }

    static dimn_t start_of_degree(const Basis& basis, deg_t deg) noexcept
    { return basis.start_of_degree(deg); }
    static dimn_t size(const Basis& basis, deg_t deg) noexcept
    { return basis.size(static_cast<int>(deg)); }
    static std::pair<dimn_t, deg_t> get_next_dimension(const Basis& basis, dimn_t dim, deg_t hint=0)
    {
        const auto& sizes = basis.sizes();
        const auto begin = sizes.begin();
        const auto end = sizes.end();
        auto it = std::lower_bound(begin, end, dim ,std::less_equal<>());
        if (it == end) {
            return {max_dimension(basis), 0};
        }
        return {*it, static_cast<deg_t>(it - begin)};
    }

    using key_ordering = std::less<key_type>;
    using kv_ordering = dtl::key_value_ordering<key_type, key_ordering>;

    static void advance_key(const Basis& basis, key_type& key)
    {
        dtl::advance_key_helper<Basis>::advance_key(basis, key);
    }

    static key_type first_key(const Basis& basis) { return basis.first_key(); }

};


template <typename Basis1, typename Basis2>
struct is_basis_compatible : std::false_type
{};



} // namespace alg

#endif //LIBALGEBRA_LITE_BASIS_TRAITS_H
