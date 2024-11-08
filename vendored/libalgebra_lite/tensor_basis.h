//
// Created by user on 25/07/22.
//

#ifndef LIBALGEBRA_LITE_TENSOR_BASIS_H
#define LIBALGEBRA_LITE_TENSOR_BASIS_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <cassert>
#include <iosfwd>
#include <string>
#include <vector>

#include "basis_traits.h"
#include "index_key.h"
#include "registry.h"

namespace lal {

class LIBALGEBRA_LITE_EXPORT tensor_basis
{
    deg_t m_width;
    deg_t m_depth;

    std::vector<dimn_t> m_powers;
    std::vector<dimn_t> m_sizes;

public:
    using key_type = index_key<>;
    using degree_tag LAL_UNUSED = with_degree_tag;

    tensor_basis(deg_t width, deg_t depth);

    deg_t width() const noexcept { return m_width; }
    deg_t depth() const noexcept { return m_depth; }

    static constexpr deg_t degree(const key_type& arg) noexcept
    {
        return deg_t(arg.degree());
    }

    key_type lparent(const key_type& arg) const noexcept
    {
        auto degree = arg.degree();
        if (degree == 0) { return key_type(0, 0); }
        return key_type(1, arg.index() / m_powers[degree - 1]);
    }
    key_type rparent(const key_type& arg) const noexcept
    {
        auto degree = arg.degree();
        if (degree == 0) { return key_type(0, 0); }
        return key_type(degree - 1, arg.index() % m_powers[degree - 1]);
    }

    pair<key_type, key_type> parents(const key_type& arg) const noexcept
    {
        auto degree = arg.degree();
        if (degree == 0) { return {key_type(0, 0), key_type(0, 0)}; }
        auto tmp = arg.index();
        auto letter = tmp / m_powers[degree - 1];
        return {key_type(1, letter),
                key_type(degree - 1, tmp - letter * m_width)};
    }

    std::string key_to_string(const key_type&) const;
    std::ostream& print_key(std::ostream&, const key_type&) const;

    static key_type key_of_letter(let_t letter) noexcept
    {
        return key_type(1, letter - 1);
    }
    let_t first_letter(const key_type& arg) const noexcept
    {
        return let_t(lparent(arg).index() + 1);
    }
    let_t to_letter(const key_type& arg) const noexcept {
        assert(arg.degree() == 1);
        return let_t(arg.index() + 1);
    }
    static bool letter(const key_type& arg) noexcept
    {
        return arg.degree() == 1;
    }

    dimn_t start_of_degree(deg_t deg) const noexcept
    {
        if (deg == 0) {
            return 0;
        } else {
            return m_sizes[deg - 1];
        }
    }
    dimn_t size_of_degree(deg_t deg) const noexcept;
    dimn_t size(int i) const noexcept
    {
        if (i >= 0) {
            return m_sizes[i];
        } else {
            return m_sizes[m_depth];
        }
    }

    const std::vector<dimn_t>& sizes() const noexcept { return m_sizes; }

    const std::vector<dimn_t>& powers() const noexcept { return m_powers; }

    dimn_t key_to_index(key_type arg) const noexcept;
    key_type index_to_key(dimn_t arg) const noexcept;

    void advance_key(key_type& key) const noexcept;

    dimn_t reverse_idx(deg_t degree, dimn_t idx) const noexcept
    {
        dimn_t result = 0;
        for (deg_t i = 0; i < degree; ++i) {
            result *= m_width;
            auto tmp = idx;
            idx /= m_width;
            result += tmp - idx * m_width;
        }
        return result;
    }

    key_type reverse_key(key_type arg) const noexcept
    {
        auto degree = arg.degree();
        auto idx = arg.index();
        auto result_idx = reverse_idx(degree, idx);
        return key_type{degree, result_idx};
    }
};

LAL_EXPORT_TEMPLATE_CLASS(basis_registry, tensor_basis)

}// namespace lal

#endif// LIBALGEBRA_LITE_TENSOR_BASIS_H
