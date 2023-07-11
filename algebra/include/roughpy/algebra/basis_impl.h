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

#ifndef ROUGHPY_ALGEBRA_BASIS_IMPL_H_
#define ROUGHPY_ALGEBRA_BASIS_IMPL_H_

#include "basis.h"

#include <roughpy/core/traits.h>

#include "basis_info.h"

namespace rpy {
namespace algebra {
namespace dtl {

/**
 * @brief Primary implementation of the basis interface
 * @tparam T Impementation Basis type
 * @tparam Interfaces interfaces that must be implemented
 */
template <typename T, typename PrimaryInterface>
class BasisImplementation : public PrimaryInterface
{
protected:
    using basis_traits = BasisInfo<Basis<PrimaryInterface>, T>;
    using basis_storage_t = typename basis_traits::storage_t;

    basis_storage_t m_impl;

public:
    using typename PrimaryInterface::key_type;

    template <typename... Args>
    explicit BasisImplementation(Args&&... args)
        : m_impl(basis_traits::construct(std::forward<Args>(args)...))
    {}

    string key_to_string(const typename PrimaryInterface::key_type& key
    ) const override;
    dimn_t dimension() const noexcept override;
};

template <typename T, typename Derived, typename Base>
class OrderedBasisImplementationMixin : public Base
{
protected:
    using Base::m_impl;
    using typename Base::basis_traits;

public:
    using Base::Base;
    using key_type = typename Derived::key_type;

    key_type index_to_key(dimn_t index) const override;
    dimn_t key_to_index(const key_type& key) const override;
};

template <typename T, typename Derived, typename Base>
class WordLikeBasisImplementationMixin : public Base
{
protected:
    using Base::m_impl;
    using typename Base::basis_traits;

public:
    using Base::Base;
    using key_type = typename Derived::key_type;

    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    deg_t degree(const key_type& key) const noexcept override;
    deg_t size(deg_t degree) const noexcept override;
    let_t first_letter(const key_type& key) const noexcept override;
    dimn_t start_of_degree(deg_t degree) const noexcept override;
    pair<optional<key_type>, optional<key_type>> parents(const key_type& key
    ) const override;
    key_type key_of_letter(let_t letter) const noexcept override;
    bool letter(const key_type& key) const override;
};

template <typename T, typename PrimaryInterface>
string BasisImplementation<T, PrimaryInterface>::key_to_string(
        const typename PrimaryInterface::key_type& key
) const
{
    return basis_traits::key_to_string(m_impl, key);
}
template <typename T, typename PrimaryInterface>
dimn_t BasisImplementation<T, PrimaryInterface>::dimension() const noexcept
{
    return basis_traits::dimension(m_impl);
}

template <typename T, typename Derived, typename Base>
typename Derived::key_type
OrderedBasisImplementationMixin<T, Derived, Base>::index_to_key(dimn_t index
) const
{
    return basis_traits::index_to_key(m_impl, index);
}
template <typename T, typename Derived, typename Base>
dimn_t OrderedBasisImplementationMixin<T, Derived, Base>::key_to_index(
        const key_type& key
) const
{
    return basis_traits::key_to_index(m_impl, key);
}

template <typename T, typename Derived, typename Base>
deg_t WordLikeBasisImplementationMixin<T, Derived, Base>::width() const noexcept
{
    return basis_traits::width(m_impl);
}
template <typename T, typename Derived, typename Base>
deg_t WordLikeBasisImplementationMixin<T, Derived, Base>::depth() const noexcept
{
    return basis_traits::depth(m_impl);
}
template <typename T, typename Derived, typename Base>
deg_t WordLikeBasisImplementationMixin<T, Derived, Base>::degree(
        const key_type& key
) const noexcept
{
    return basis_traits::degree(m_impl, key);
}
template <typename T, typename Derived, typename Base>
deg_t WordLikeBasisImplementationMixin<T, Derived, Base>::size(deg_t degree
) const noexcept
{
    return basis_traits::size(m_impl, degree);
}
template <typename T, typename Derived, typename Base>
let_t WordLikeBasisImplementationMixin<T, Derived, Base>::first_letter(
        const key_type& key
) const noexcept
{
    return basis_traits::first_letter(m_impl, key);
}
template <typename T, typename Derived, typename Base>
dimn_t
WordLikeBasisImplementationMixin<T, Derived, Base>::start_of_degree(deg_t degree
) const noexcept
{
    return basis_traits::start_of_degree(m_impl, degree);
}
template <typename T, typename Derived, typename Base>
pair<optional<typename Derived::key_type>, optional<typename Derived::key_type>>
WordLikeBasisImplementationMixin<T, Derived, Base>::parents(const key_type& key
) const
{
    return basis_traits::parents(m_impl, key);
}
template <typename T, typename Derived, typename Base>
typename Derived::key_type
WordLikeBasisImplementationMixin<T, Derived, Base>::key_of_letter(let_t letter
) const noexcept
{
    return basis_traits::key_of_letter(m_impl, letter);
}
template <typename T, typename Derived, typename Base>
bool WordLikeBasisImplementationMixin<T, Derived, Base>::letter(
        const key_type& key
) const
{
    return basis_traits::letter(m_impl, key);
}

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_IMPL_H_
