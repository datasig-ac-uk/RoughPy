#ifndef ROUGHPY_ALGEBRA_BASIS_H_
#define ROUGHPY_ALGEBRA_BASIS_H_

#include "algebra_fwd.h"

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <roughpy/core/traits.h>


namespace rpy {
namespace algebra {




namespace dtl {
template <typename KeyType, typename Derived>
class StandardBasisMixin;

template <typename KeyType, typename Derived>
class StandardBasisImplementationMixin;

}

template <typename KeyType = rpy::key_type>
class BasisInterface {
public:
    using key_type = KeyType;

    template <typename D>
    using mixin_class = dtl::StandardBasisMixin<KeyType, D>;

    template <typename D>
    using impl_mixin_class = dtl::StandardBasisImplementationMixin<KeyType, D>;

    virtual ~BasisInterface() = default;

    virtual std::string key_to_string(const key_type& key) const = 0;

    virtual dimn_t dimension() const noexcept = 0;
};

namespace dtl {
template <typename KeyType, typename Derived>
class OrderedBasisMixin;

template <typename KeyType, typename Derived>
class OrderedBasisImplementationMixin;

}

template <typename KeyType = rpy::key_type>
class OrderedBasisInterface {

public:
    using key_type = KeyType;

    template <typename D>
    using mixin_class = dtl::OrderedBasisMixin<KeyType, D>;

    template <typename D>
    using impl_mixin_class = dtl::OrderedBasisImplementationMixin<KeyType, D>;

    virtual ~OrderedBasisInterface() = default;

    virtual key_type index_to_key(dimn_t index) const = 0;
    virtual dimn_t key_to_index(const key_type &key) const = 0;
};

namespace dtl {
template <typename KeyType, typename Derived>
class WordLikeBasisMixin;

template <typename KeyType, typename Derived>
class WordLikeBasisImplementationMixin;

}

template <typename KeyType = rpy::key_type>
class WordLikeBasisInterface {

public:
    using key_type = KeyType;
    template <typename D>
    using mixin_class = dtl::WordLikeBasisMixin<KeyType, D>;

    template <typename D>
    using impl_mixin_class = dtl::WordLikeBasisImplementationMixin<KeyType, D>;

    virtual ~WordLikeBasisInterface() = default;
    virtual deg_t width() const noexcept = 0;
    virtual deg_t depth() const noexcept = 0;
    virtual deg_t degree(const KeyType& key) const noexcept = 0;
    virtual deg_t size(deg_t degree) const noexcept = 0;
    virtual let_t first_letter(const KeyType &key) const noexcept = 0;

    virtual dimn_t start_of_degree(deg_t degree) const noexcept = 0;
    virtual pair<optional<KeyType>, optional<KeyType>>
    parents(const KeyType &key) const = 0;
    optional<KeyType> lparent(const KeyType &key) const {
        return parents(key).first;
    }
    optional<KeyType> rparent(const KeyType &key) const {
        return parents(key).second;
    }
    virtual key_type key_of_letter(let_t letter) const noexcept = 0;
    virtual bool letter(const KeyType &key) const = 0;
};

namespace dtl {
template <typename T, typename... Interfaces>
class BasisImplementation;
}



template <typename... Interfaces>
class Basis : Interfaces::template mixin_class<Basis<Interfaces...>>... {
    using basis_interface = traits::select_first_t<Interfaces...>;
    static_assert(
        traits::is_base_of<
            BasisInterface<typename basis_interface::key_type>,
            basis_interface>::value,
        "Primary template must be an instance of BasisInterface");

    boost::intrusive_ptr<const basis_interface> p_impl;

public:
    using key_type = typename basis_interface::key_type;

    template <typename B>
    explicit Basis(const B *basis)
        : p_impl(new dtl::BasisImplementation<B, Interfaces...>(basis)) {}

    template <typename B>
    explicit Basis(const B& basis)
        : p_impl(new dtl::BasisImplementation<B, Interfaces...>(basis))
    {}

    const basis_interface &instance() const noexcept { return *p_impl; }
};

namespace dtl {

template <typename KeyType, typename Derived>
class StandardBasisMixin {
public:
    std::string key_to_string(const KeyType &key) const noexcept {
        return static_cast<const Derived *>(this)->instance().key_to_string(key);
    }

    dimn_t dimension() const noexcept {
        return static_cast<const Derived *>(this)->instance().dimension();
    }
};

template <typename KeyType, typename Derived>
class OrderedBasisMixin {
public:
    KeyType index_to_key(dimn_t index) const {
        return static_cast<const Derived *>(this)->instance().index_to_key(index);
    }

    dimn_t key_to_index(const KeyType &key) const {
        return static_cast<const Derived *>(this)->instance().key_to_index(key);
    }
};

template <typename KeyType, typename Derived>
class WordLikeBasisMixin {
public:
    deg_t width() const noexcept {
        return static_cast<const Derived*>(this)->instance().width();
    }
    deg_t depth() const noexcept {
        return static_cast<const Derived *>(this)->instance().depth();
    }
    deg_t degree(const KeyType& key) const noexcept {
        return static_cast<const Derived *>(this)->instance().degree(key);
    }
    deg_t size(deg_t degree) const noexcept {
        return static_cast<const Derived*>(this)->instance().size(degree);
    }
    let_t first_letter(const KeyType &key) const noexcept {
        return static_cast<const Derived *>(this)->instance().first_letter(key);
    }
    dimn_t start_of_degree(deg_t degree) const noexcept {
        return static_cast<const Derived *>(this)->instance().start_of_degree(degree);
    }
    pair<optional<KeyType>, optional<KeyType>> parents(const KeyType &key) const {
        return static_cast<const Derived *>(this)->instance().parents(key);
    }
    optional<KeyType> lparent(const KeyType &key) const {
        static_cast<const Derived*>(this)->instance().lparent(key);
    }
    optional<KeyType> rparent(const KeyType &key) const {
        static_cast<const Derived*>(this)->instance().rparent(key);
    }
    key_type key_of_letter(let_t letter) const noexcept {
        return static_cast<const Derived *>(this)->instance().key_of_letter(letter);
    }
    bool letter(const KeyType &key) const {
        return static_cast<const Derived *>(this)->instance().letter(key);
    }



};
}// namespace dtl

extern template class ROUGHPY_ALGEBRA_EXPORT BasisInterface<>;
extern template class ROUGHPY_ALGEBRA_EXPORT OrderedBasisInterface<>;
extern template class ROUGHPY_ALGEBRA_EXPORT WordLikeBasisInterface<>;
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_
