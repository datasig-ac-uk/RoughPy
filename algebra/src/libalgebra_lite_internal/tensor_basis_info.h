// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H

#include <libalgebra_lite/tensor_basis.h>

#include <roughpy/core/types.h>
#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/basis_impl.h>
#include <roughpy/algebra/basis_info.h>
#include <roughpy/algebra/tensor_basis.h>


namespace rpy {
namespace algebra {

template <>
struct BasisInfo<TensorBasis, lal::tensor_basis> {
    /// Type that should be stored in the basis implementation
    using storage_t = const lal::tensor_basis*;

    /// The key type that is handled internally in roughpy
    using our_key_type = typename TensorBasis::key_type;

    /// The key type that is handled internally in the implementation
    using impl_key_type = typename lal::tensor_basis::key_type;

    /// Method of constructing a storage_t from arguments
    static storage_t construct(storage_t arg) { return arg; }

    /*
     * For methods that require access to the basis, the first
     * argument should be a storage_t or const reference thereof
     * and the remaining arguments should be the ones required
     * for each method, using rpy types as appropriate.
     */

    // The conversion methods are required, they are used in the trait
    // and in the algebra wrapper
    /// Conversion from impl_key_type to our_key_type
    static our_key_type
    convert_from_impl(storage_t basis, const impl_key_type& arg)
    {
        // The default is to take the index of arg as the return
        return basis->key_to_index(arg);
    }

    /// Conversion from our_key_type to impl_key_type
    static impl_key_type
    convert_to_impl(storage_t basis, const our_key_type& arg)
    {
        // Default is to treat rpy keys as the index of impl keys
        return basis->index_to_key(arg);
    }

    /*
     * A standard basis requires as a minimum:
     *      - key_to_string
     *      - dimension
     */

    /// Generate a string representation of the key
    static string key_to_string(storage_t basis, const our_key_type& key)
    {
        return basis->key_to_string(convert_to_impl(basis, key));
    }

    /// Get the dimension of the span of this basis
    static dimn_t dimension(storage_t basis) noexcept
    {
        return basis->size(-1);
    }

    /*
     * An ordered basis requires as a minimum:
     *      - index_to_key
     *      - key_to_index
     */

    /// Get the key in the basis total order whose index is given
    static our_key_type index_to_key(storage_t basis, dimn_t index)
    {
        return convert_from_impl(basis, basis->index_to_key(index));
    }

    /// Get the index of key in the basis total order
    static dimn_t key_to_index(storage_t basis, const our_key_type& key)
    {
        return basis->key_to_index(convert_to_impl(basis, key));
    }

    /*
     * A word-like basis requires at a mimimum:
     *      - width
     *      - depth
     *      - degree
     *      - size
     *      - start_of_degree
     *      - parents
     *      - first_letter
     *      - key_of_letter
     *      - letter
     */

    /// Get the size of the alphabet that defines the basis
    static deg_t width(storage_t basis) { return basis->width(); }

    /// Get the maximum word length for basis elements
    static deg_t depth(storage_t basis) { return basis->depth(); }

    /// Get the length of a key as a word
    static deg_t degree(storage_t basis, const our_key_type& key)
    {
        return basis->degree(convert_to_impl(basis, key));
    }

    /// Get the size of the subspace of elements with degree at most given
    static dimn_t size(storage_t basis, deg_t degree)
    {
        return basis->size(degree);
    }

    /// Get the index at which the elements of given degree start in the
    /// basis total order
    static dimn_t start_of_degree(storage_t basis, deg_t degree)
    {
        return basis->start_of_degree(degree);
    }

    /// Get the parents of a key according to the basis composition
    static pair<optional<our_key_type>, optional<our_key_type>>
    parents(storage_t basis, const our_key_type& key)
    {
        auto tmpkey = convert_to_impl(basis, key);
        return {convert_from_impl(basis, basis->lparent(tmpkey)),
                convert_from_impl(basis, basis->rparent(tmpkey))};
    }

    static optional<our_key_type> child(storage_t basis, const our_key_type& lparent, const our_key_type& rparent)
    {
        const auto left = convert_to_impl(basis, lparent);
        const auto right = convert_to_impl(basis, rparent);

        const auto ldegree = left.degree();
        const auto rdegree = right.degree();

        optional<our_key_type> out {};

        const auto degree = static_cast<deg_t>(ldegree + rdegree);
        if (degree <= basis->depth()) {
            const auto shift = basis->powers()[rdegree];
            const auto idx = left.index() * shift + right.index();

            out = convert_from_impl(basis, impl_key_type {degree, idx});
        }

        return out;
    }

    /// Get the first letter of the key as a word
    static let_t first_letter(storage_t basis, const our_key_type& key)
    {
        return basis->first_letter(convert_to_impl(basis, key));
    }

    static let_t to_letter(storage_t basis, const our_key_type& key) {
        return basis->to_letter(convert_to_impl(basis, key));
    }

    /// Get the key type that represents letter
    static our_key_type key_of_letter(storage_t basis, let_t letter)
    {
        return convert_from_impl(
                basis, lal::tensor_basis::key_of_letter(letter)
        );
    }

    /// Determine whether a key represents a single letter
    static bool letter(storage_t basis, const our_key_type& key)
    {
        return lal::tensor_basis::letter(convert_to_impl(basis, key));
    }


    static bool are_same(storage_t basis1, storage_t basis2) noexcept { return basis1 == basis2; }
};
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_BASIS_INFO_H
