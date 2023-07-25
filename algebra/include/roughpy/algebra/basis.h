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

#ifndef ROUGHPY_ALGEBRA_BASIS_H_
#define ROUGHPY_ALGEBRA_BASIS_H_

#include "algebra_fwd.h"

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <roughpy/core/traits.h>

namespace rpy {
namespace algebra {

namespace dtl {
template <typename T, typename PrimaryInterface>
class BasisImplementation;
}

template <typename Derived, typename KeyType = rpy::key_type>
class BasisInterface
    : public boost::intrusive_ref_counter<BasisInterface<KeyType>>
{
public:
    using key_type = KeyType;

    template <typename T>
    using impl_t = dtl::BasisImplementation<T, Derived>;

    using mixin_t = void;

    virtual ~BasisInterface() = default;

    RPY_NO_DISCARD
    virtual string key_to_string(const key_type& key) const = 0;

    RPY_NO_DISCARD
    virtual dimn_t dimension() const noexcept = 0;
};

namespace dtl {
template <typename Derived, typename Base>
class OrderedBasisMixin;

template <typename T, typename Derived, typename Base>
class OrderedBasisImplementationMixin;

}// namespace dtl

template <typename Derived, typename Base>
class OrderedBasisInterface : public void_or_base<Base>
{

public:
    using typename Base::key_type;

    using mixin_t = dtl::OrderedBasisMixin<Derived, typename Base::mixin_t>;

    template <typename T>
    using impl_t = dtl::OrderedBasisImplementationMixin<
            T, Derived, typename Base::template impl_t<T>>;

    virtual ~OrderedBasisInterface() = default;

    RPY_NO_DISCARD
    virtual key_type index_to_key(dimn_t index) const = 0;
    RPY_NO_DISCARD
    virtual dimn_t key_to_index(const key_type& key) const = 0;
};

namespace dtl {
template <typename Derived, typename Base>
class WordLikeBasisMixin;

template <typename T, typename Derived, typename Base>
class WordLikeBasisImplementationMixin;

}// namespace dtl

template <typename Derived, typename Base>
class WordLikeBasisInterface : public void_or_base<Base>
{
public:
    using typename Base::key_type;

    using mixin_t = dtl::WordLikeBasisMixin<Derived, typename Base::mixin_t>;

    template <typename T>
    using impl_t = dtl::WordLikeBasisImplementationMixin<
            T, Derived, typename Base::template impl_t<T>>;

    RPY_NO_DISCARD
    virtual deg_t width() const noexcept = 0;
    RPY_NO_DISCARD
    virtual deg_t depth() const noexcept = 0;
    RPY_NO_DISCARD
    virtual deg_t degree(const key_type& key) const noexcept = 0;
    RPY_NO_DISCARD
    virtual deg_t size(deg_t degree) const noexcept = 0;
    RPY_NO_DISCARD
    virtual let_t first_letter(const key_type& key) const noexcept = 0;

    RPY_NO_DISCARD
    virtual dimn_t start_of_degree(deg_t degree) const noexcept = 0;
    RPY_NO_DISCARD
    virtual pair<optional<key_type>, optional<key_type>>
    parents(const key_type& key) const = 0;
    RPY_NO_DISCARD
    optional<key_type> lparent(const key_type& key) const
    {
        return parents(key).first;
    }
    RPY_NO_DISCARD
    optional<key_type> rparent(const key_type& key) const
    {
        return parents(key).second;
    }
    RPY_NO_DISCARD
    virtual key_type key_of_letter(let_t letter) const noexcept = 0;
    RPY_NO_DISCARD
    virtual bool letter(const key_type& key) const = 0;
};

template <typename PrimaryInterface>
class Basis : public PrimaryInterface::mixin_t
{
    using basis_interface = PrimaryInterface;
    static_assert(
            is_base_of<
                    BasisInterface<
                            PrimaryInterface,
                            typename basis_interface::key_type>,
                    basis_interface>::value,
            "Primary template must be an instance of BasisInterface"
    );

    boost::intrusive_ptr<const basis_interface> p_impl;

public:
    using key_type = typename basis_interface::key_type;

    template <typename B>
    explicit Basis(const B* basis)
        : p_impl(new typename PrimaryInterface::template impl_t<B>(basis))
    {}

    //    template <typename B>
    //    explicit Basis(const B& basis)
    //        : p_impl(new typename PrimaryInterface::template impl_t<B>(basis))
    //    {}

    Basis() = default;
    Basis(const Basis&) = default;
    Basis(Basis&&) noexcept = default;

    RPY_NO_UBSAN
    ~Basis() = default;

    Basis& operator=(const Basis&) = default;
    Basis& operator=(Basis&&) noexcept = default;

    RPY_NO_DISCARD
    const basis_interface& instance() const noexcept { return *p_impl; }

    RPY_NO_DISCARD
    constexpr operator bool() const noexcept { return static_cast<bool>(p_impl); }

    RPY_NO_DISCARD
    string key_to_string(const key_type& key) const noexcept
    {
        return instance().key_to_string(key);
    }

    RPY_NO_DISCARD
    dimn_t dimension() const noexcept { return instance().dimension(); }


    friend bool operator==(const Basis& left, const Basis& right) noexcept
    {
        return left.p_impl == right.p_impl;
    }

    friend bool operator!=(const Basis& left, const Basis& right) noexcept
    {
        return left.p_impl != right.p_impl;
    }

};



namespace dtl {
template <
        typename Derived, typename KeyType,
        template <typename, typename> class... Interfaces>
struct make_basis_interface_impl;
}

template <
        typename Derived, typename KeyType,
        template <typename, typename> class... Interfaces>
using make_basis_interface = typename dtl::make_basis_interface_impl<
        Derived, KeyType, Interfaces...>::type;

namespace dtl {

template <typename Derived, typename Base>
class OrderedBasisMixin : public Base
{

    RPY_NO_DISCARD
    const Derived& instance() const noexcept
    {
        return static_cast<const Basis<Derived>*>(this)->instance();
    }

public:
    using key_type = typename Derived::key_type;

    RPY_NO_DISCARD
    key_type index_to_key(dimn_t index) const
    {
        return instance().index_to_key(index);
    }

    RPY_NO_DISCARD
    dimn_t key_to_index(const key_type& key) const
    {
        return instance().key_to_index(key);
    }
};

template <typename Derived, typename Base>
class WordLikeBasisMixin
{

    RPY_NO_DISCARD
    const Derived& instance() const noexcept
    {
        return static_cast<const Basis<Derived>*>(this)->instance();
    }

public:
    using key_type = typename Derived::key_type;

    RPY_NO_DISCARD
    deg_t width() const noexcept { return instance().width(); }
    RPY_NO_DISCARD
    deg_t depth() const noexcept { return instance().depth(); }
    RPY_NO_DISCARD
    deg_t degree(const key_type& key) const noexcept
    {
        return instance().degree(key);
    }
    RPY_NO_DISCARD
    deg_t size(deg_t degree) const noexcept { return instance().size(degree); }
    RPY_NO_DISCARD
    let_t first_letter(const key_type& key) const noexcept
    {
        return instance().first_letter(key);
    }
    RPY_NO_DISCARD
    dimn_t start_of_degree(deg_t degree) const noexcept
    {
        return instance().start_of_degree(degree);
    }
    RPY_NO_DISCARD
    pair<optional<key_type>, optional<key_type>> parents(const key_type& key
    ) const
    {
        return instance().parents(key);
    }
    RPY_NO_DISCARD
    optional<key_type> lparent(const key_type& key) const
    {
        return instance().lparent(key);
    }
    RPY_NO_DISCARD
    optional<key_type> rparent(const key_type& key) const
    {
        return instance().rparent(key);
    }
    RPY_NO_DISCARD
    key_type key_of_letter(let_t letter) const noexcept
    {
        return instance().key_of_letter(letter);
    }
    RPY_NO_DISCARD
    bool letter(const key_type& key) const { return instance().letter(key); }
};

template <
        typename Derived, typename KeyType,
        template <typename, typename> class FirstInterface,
        template <typename, typename> class... Interfaces>
struct make_basis_interface_impl<
        Derived, KeyType, FirstInterface, Interfaces...> {
    using next_t = make_basis_interface_impl<Derived, KeyType, Interfaces...>;
    using type = FirstInterface<Derived, typename next_t::type>;
};

template <typename Derived, typename KeyType>
struct make_basis_interface_impl<Derived, KeyType> {
    using type = BasisInterface<Derived, KeyType>;
};

}// namespace dtl

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_
