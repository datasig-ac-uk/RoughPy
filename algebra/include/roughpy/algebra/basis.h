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

#include <roughpy/core/hash.h>
#include <roughpy/core/smart_ptr.h>

#include <roughpy/core/traits.h>

namespace rpy {
namespace algebra {



template <typename KeyType = rpy::key_type>
class BasisInterface
    : public mem::RcBase<BasisInterface<KeyType>>
{
public:
    using key_type = KeyType;


    using base_interface_t = BasisInterface;

    using mixin_t = void;

    virtual ~BasisInterface() = default;

    RPY_NO_DISCARD virtual string key_to_string(const key_type& key) const = 0;

    RPY_NO_DISCARD virtual dimn_t dimension() const noexcept = 0;

    RPY_NO_DISCARD virtual bool are_same(const BasisInterface &other) const noexcept = 0;

    
    RPY_NO_DISCARD virtual key_type index_to_key(dimn_t index) const = 0;
    RPY_NO_DISCARD virtual dimn_t key_to_index(const key_type& key) const = 0;

    RPY_NO_DISCARD virtual deg_t width() const noexcept = 0;
    RPY_NO_DISCARD virtual deg_t depth() const noexcept = 0;
    RPY_NO_DISCARD virtual deg_t degree(const key_type& key) const noexcept = 0;
    RPY_NO_DISCARD virtual deg_t size(deg_t degree) const noexcept = 0;
    RPY_NO_DISCARD virtual let_t first_letter(const key_type& key
    ) const noexcept
            = 0;
    RPY_NO_DISCARD virtual let_t to_letter(const key_type& key) const noexcept
            = 0;

    RPY_NO_DISCARD virtual dimn_t start_of_degree(deg_t degree) const noexcept
            = 0;
    RPY_NO_DISCARD virtual pair<optional<key_type>, optional<key_type>>
    parents(const key_type& key) const = 0;
    RPY_NO_DISCARD optional<key_type> lparent(const key_type& key) const
    {
        return parents(key).first;
    }
    RPY_NO_DISCARD optional<key_type> rparent(const key_type& key) const
    {
        return parents(key).second;
    }

    RPY_NO_DISCARD virtual optional<key_type>
    child(const key_type& lparent, const key_type& rparent) const = 0;

    RPY_NO_DISCARD virtual key_type key_of_letter(let_t letter) const noexcept
            = 0;
    RPY_NO_DISCARD virtual bool letter(const key_type& key) const = 0;
    
};


namespace dtl {

template <typename T, typename Interface>
class BasisImplementation;

}


template <typename Interface>
/**
 *
 */
class Basis 
{
    Rc<const Interface> p_impl;

public:
    using key_type = typename Interface::key_type;

    template <typename B>
    explicit Basis(const B* basis)
        : p_impl(new dtl::BasisImplementation<B, Interface>(basis))
    {}

    //    template <typename B>
    //    explicit Basis(const B& basis)
    //        : p_impl(new typename PrimaryInterface::template impl_t<B>(basis))
    //    {}

    Basis() = default;
    Basis(const Basis&) = default;
    Basis(Basis&&) noexcept = default;

    ~Basis() = default;

    Basis& operator=(const Basis&) = default;
    Basis& operator=(Basis&&) noexcept = default;

    RPY_NO_DISCARD const Interface& instance() const noexcept
    {
        return *p_impl;
    }

    RPY_NO_DISCARD constexpr operator bool() const noexcept
    {
        return static_cast<bool>(p_impl);
    }

    RPY_NO_DISCARD string key_to_string(const key_type& key) const noexcept
    {
        return instance().key_to_string(key);
    }

    RPY_NO_DISCARD dimn_t dimension() const noexcept
    {
        return instance().dimension();
    }

    friend bool operator==(const Basis& left, const Basis& right) noexcept
    {
        if (left.p_impl == right.p_impl) {
            return true;
        }

        return left.p_impl->are_same(*right.p_impl);
    }

    friend bool operator!=(const Basis& left, const Basis& right) noexcept
    {
        return !operator==(left, right);
    }

    friend hash_t hash_value(const Basis& basis) noexcept
    {
        return hash_value(basis.p_impl);
    }

    RPY_NO_DISCARD key_type index_to_key(dimn_t index) const
    {
        return p_impl->index_to_key(index);
    }

    RPY_NO_DISCARD dimn_t key_to_index(const key_type& key) const
    {
        return p_impl->key_to_index(key);
    }
 RPY_NO_DISCARD deg_t width() const noexcept { return p_impl->width(); }
    RPY_NO_DISCARD deg_t depth() const noexcept { return p_impl->depth(); }
    RPY_NO_DISCARD deg_t degree(const key_type& key) const noexcept
    {
        return p_impl->degree(key);
    }
    RPY_NO_DISCARD deg_t size(deg_t degree) const noexcept
    {
        return p_impl->size(degree);
    }
    RPY_NO_DISCARD let_t first_letter(const key_type& key) const noexcept
    {
        return p_impl->first_letter(key);
    }
    RPY_NO_DISCARD let_t to_letter(const key_type& key) const noexcept
    {
        return p_impl->to_letter(key);
    }
    RPY_NO_DISCARD dimn_t start_of_degree(deg_t degree) const noexcept
    {
        return p_impl->start_of_degree(degree);
    }
    RPY_NO_DISCARD pair<optional<key_type>, optional<key_type>>
    parents(const key_type& key) const
    {
        return p_impl->parents(key);
    }
    RPY_NO_DISCARD optional<key_type> lparent(const key_type& key) const
    {
        return p_impl->lparent(key);
    }
    RPY_NO_DISCARD optional<key_type> rparent(const key_type& key) const
    {
        return p_impl->rparent(key);
    }
    RPY_NO_DISCARD optional<key_type>
    child(const key_type& lparent, const key_type& rparent) const
    {
        return p_impl->child(lparent, rparent);
    }

    RPY_NO_DISCARD key_type key_of_letter(let_t letter) const noexcept
    {
        return p_impl->key_of_letter(letter);
    }
    RPY_NO_DISCARD bool letter(const key_type& key) const
    {
        return p_impl->letter(key);
    }

};



}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_
