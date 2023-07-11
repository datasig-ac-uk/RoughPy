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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_IMPL_H_

#include "algebra_info.h"
#include "algebra_iterator.h"

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_traits.h>

namespace rpy {
namespace algebra {

template <typename Iter>
struct iterator_helper_trait {
    static auto key(const Iter& it) noexcept -> decltype(it->first)
    {
        return it->first;
    }
    static auto value(const Iter& it) noexcept -> decltype(it->second)
    {
        return it->second;
    }
};

template <typename Algebra, typename RealBasis, typename Iter>
class AlgebraIteratorImplementation : public AlgebraIteratorInterface<Algebra>
{
    Iter m_iter;
    const RealBasis* p_basis;
    using interface_type = AlgebraIteratorInterface<Algebra>;

    using btraits = BasisInfo<typename Algebra::basis_type, RealBasis>;
    using itraits = iterator_helper_trait<Iter>;

public:
    using basis_type = typename Algebra::basis_type;
    using key_type = typename Algebra::key_type;

    AlgebraIteratorImplementation(Iter iter, const RealBasis* basis)
        : interface_type(basis_type(basis)), m_iter(std::move(iter)),
          p_basis(basis)
    {}

    key_type key() const noexcept override
    {
        return btraits::convert_from_impl(p_basis, itraits::key(m_iter));
    }
    scalars::Scalar value() const noexcept override
    {
        using trait
                = scalars::scalar_type_trait<decltype(itraits::value(m_iter))>;
        return trait::make(itraits::value(m_iter));
    }
    std::shared_ptr<interface_type> clone() const override
    {
        return std::shared_ptr<interface_type>(
                new AlgebraIteratorImplementation(m_iter, p_basis)
        );
    }
    void advance() override { ++m_iter; }
    bool equals(const interface_type& other) const noexcept override
    {
        // We only get here if the vector of both iterators is the same
        return m_iter
                == static_cast<const AlgebraIteratorImplementation&>(other)
                           .m_iter;
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_IMPL_H_
