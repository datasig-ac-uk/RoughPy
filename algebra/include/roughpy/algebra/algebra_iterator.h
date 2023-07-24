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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_

#include "algebra_fwd.h"
#include "basis.h"

#include <roughpy/scalars/scalar.h>

#include <memory>

namespace rpy {
namespace algebra {

template <typename Algebra>
class AlgebraIteratorInterface
{
    typename Algebra::basis_type m_basis;

    using basis_type = typename Algebra::basis_type;

protected:
    explicit AlgebraIteratorInterface(basis_type basis)
        : m_basis(std::move(basis))
    {}

public:
    virtual ~AlgebraIteratorInterface() = default;

    using key_type = typename basis_type::key_type;

    RPY_NO_DISCARD
    const basis_type& basis() const { return m_basis; };
    RPY_NO_DISCARD
    virtual key_type key() const noexcept = 0;
    RPY_NO_DISCARD
    virtual scalars::Scalar value() const noexcept = 0;

    virtual std::shared_ptr<AlgebraIteratorInterface> clone() const = 0;
    virtual void advance() = 0;
    RPY_NO_DISCARD
    virtual bool equals(const AlgebraIteratorInterface& other) const noexcept
            = 0;
};

template <typename Algebra>
class AlgebraIteratorItem
{
    std::shared_ptr<AlgebraIteratorInterface<Algebra>> p_interface;

public:
    using basis_type = typename Algebra::basis_type;
    using key_type = typename Algebra::key_type;

    AlgebraIteratorItem(std::shared_ptr<AlgebraIteratorInterface<Algebra>>
                                interface)
        : p_interface(std::move(interface))
    {}

    RPY_NO_DISCARD
    const basis_type& basis() const { return p_interface->basis(); }
    RPY_NO_DISCARD
    key_type key() const noexcept { return p_interface->key(); }
    RPY_NO_DISCARD
    scalars::Scalar value() const noexcept { return p_interface->value(); };

    RPY_NO_DISCARD
    AlgebraIteratorItem* operator->() noexcept { return this; }
    RPY_NO_DISCARD
    AlgebraIteratorItem& operator*() noexcept { return *this; }
};

template <typename Algebra>
class AlgebraIterator
{
public:
    using interface_type = AlgebraIteratorInterface<Algebra>;
    using value_type = AlgebraIteratorItem<Algebra>;

private:
    std::shared_ptr<interface_type> p_interface;
    std::uintptr_t m_tag;

public:
    using reference = value_type;
    using pointer = value_type;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    AlgebraIterator(
            std::shared_ptr<interface_type> interface, std::uintptr_t tag
    )
        : p_interface(std::move(interface)), m_tag(tag)
    {}

    AlgebraIterator() = default;
    AlgebraIterator(const AlgebraIterator& arg);
    AlgebraIterator(AlgebraIterator&& arg) noexcept;

    AlgebraIterator& operator=(const AlgebraIterator& arg);
    AlgebraIterator& operator=(AlgebraIterator&& arg) noexcept;

    AlgebraIterator& operator++();
    RPY_NO_DISCARD
    const AlgebraIterator operator++(int);
    RPY_NO_DISCARD
    reference operator*() const;
    RPY_NO_DISCARD
    pointer operator->() const;

    RPY_NO_DISCARD
    bool operator==(const AlgebraIterator& other) const;
    RPY_NO_DISCARD
    bool operator!=(const AlgebraIterator& other) const;
};

template <typename Algebra>
AlgebraIterator<Algebra>::AlgebraIterator(const AlgebraIterator& arg)
    : m_tag(arg.m_tag)
{
    if (arg.p_interface) { p_interface = arg.p_interface->clone(); }
}
template <typename Algebra>
AlgebraIterator<Algebra>::AlgebraIterator(AlgebraIterator&& arg) noexcept
    : p_interface(std::move(arg.p_interface)), m_tag(arg.m_tag)
{}
template <typename Algebra>
AlgebraIterator<Algebra>&
AlgebraIterator<Algebra>::operator=(const AlgebraIterator& arg)
{
    if (&arg != this) {
        if (arg.p_interface) {
            p_interface = arg.p_interface->clone();
        } else {
            p_interface.reset();
        }
    }
    m_tag = arg.m_tag;
    return *this;
}
template <typename Algebra>
AlgebraIterator<Algebra>&
AlgebraIterator<Algebra>::operator=(AlgebraIterator&& arg) noexcept
{
    if (&arg != this) { p_interface = std::move(arg.p_interface); }
    m_tag = arg.m_tag;
    return *this;
}
template <typename Algebra>
AlgebraIterator<Algebra>& AlgebraIterator<Algebra>::operator++()
{
    if (p_interface) { p_interface->advance(); }
    return *this;
}
template <typename Algebra>
const AlgebraIterator<Algebra> AlgebraIterator<Algebra>::operator++(int)
{
    AlgebraIterator result(*this);
    if (p_interface) { p_interface->advance(); }
    return result;
}
template <typename Algebra>
typename AlgebraIterator<Algebra>::reference
AlgebraIterator<Algebra>::operator*() const
{
    if (!p_interface) {
        RPY_THROW(std::runtime_error,"attempting to dereference an invalid iterator"
        );
    }
    return {p_interface};
}
template <typename Algebra>
typename AlgebraIterator<Algebra>::pointer
AlgebraIterator<Algebra>::operator->() const
{
    if (!p_interface) {
        RPY_THROW(std::runtime_error, "cannot dereference an invalid iterator");
    }
    return {p_interface};
}
template <typename Algebra>
bool AlgebraIterator<Algebra>::operator==(const AlgebraIterator& other) const
{
    if (!p_interface || !other.p_interface) { return false; }
    if (m_tag != other.m_tag) { return false; }
    return p_interface->equals(*other.p_interface);
}
template <typename Algebra>
bool AlgebraIterator<Algebra>::operator!=(const AlgebraIterator& other) const
{
    if (!p_interface || !other.p_interface) { return true; }
    if (m_tag != other.m_tag) { return true; }
    return !p_interface->equals(*other.p_interface);
}

}// namespace algebra
}// namespace rpy
#endif// ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_
