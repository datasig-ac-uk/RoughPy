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
 * @tparam T Basis type
 * @tparam Interfaces interfaces that must be implemented
 */
template <typename T, typename... Interfaces>
class BasisImplementation
    : public boost::intrusive_ref_counter<BasisInterface<typename
            traits::select_first_t<Interfaces...>::key_type>>,
      public Interfaces::template
        impl_mixin_class<BasisImplementation<T, Interfaces...>>...
{

public:
    using basis_traits = BasisInfo<Basis<Interfaces...>, T>;
    using basis_storage_t = typename basis_traits::storage_t;

    // The basis implementation itself is public (and const) so
    // the mixin classes can get their hands on it easily.
    const basis_storage_t m_impl;

    template <typename... Args>
    explicit BasisImplementation(Args&&... args)
        : m_impl(basis_traits::construct(std::forward<Args>(args)...))
    {}

};


template <typename KeyType, typename Derived>
class StandardBasisImplementationMixin : public BasisInterface<KeyType> {
public:
    std::string key_to_string(const key_type& key) const override;
    dimn_t dimension() const noexcept override;
};

template <typename KeyType, typename Derived>
class OrderedBasisImplementationMixin : public OrderedBasisInterface<KeyType> {
public:
    key_type index_to_key(dimn_t index) const override;
    dimn_t key_to_index(const key_type &key) const override;
};

template <typename KeyType, typename Derived>
class WordLikeBasisImplementationMixin : public WordLikeBasisInterface<KeyType> {

public:
    deg_t width() const noexcept override;
    deg_t depth() const noexcept override;
    deg_t degree(const KeyType& key) const noexcept override;
    deg_t size(deg_t degree) const noexcept override;
    let_t first_letter(const KeyType &key) const noexcept override;
    dimn_t start_of_degree(deg_t degree) const noexcept override;
    pair<optional<KeyType>, optional<KeyType>> parents(const KeyType &key) const override;
    key_type key_of_letter(let_t letter) const noexcept override;
    bool letter(const KeyType &key) const override;
};


template <typename KeyType, typename Derived>
std::string StandardBasisImplementationMixin<KeyType, Derived>::key_to_string(const key_type& key) const {
    return Derived::basis_traits::key_to_string(static_cast<const Derived*>(this)->m_impl, key);
}
template <typename KeyType, typename Derived>
dimn_t StandardBasisImplementationMixin<KeyType, Derived>::dimension() const noexcept {
    return Derived::basis_traits::dimension(static_cast<const Derived*>(*this)->m_impl);
}

template <typename KeyType, typename Derived>
key_type OrderedBasisImplementationMixin<KeyType, Derived>::index_to_key(dimn_t index) const {
    return Derived::basis_traits::index_to_key(
        static_cast<const Derived*>(this)->m_impl,
        index
        );
}
template <typename KeyType, typename Derived>
dimn_t OrderedBasisImplementationMixin<KeyType, Derived>::key_to_index(const key_type &key) const {
    return Derived::basis_traits::key_to_index(
        static_cast<const Derived*>(this)->m_impl,
        key
        );
}

template <typename KeyType, typename Derived>
deg_t WordLikeBasisImplementationMixin<KeyType, Derived>::width() const noexcept {
    return Derived::basis_traits::width(static_cast<const Derived*>(this)->m_impl);
}
template <typename KeyType, typename Derived>
deg_t WordLikeBasisImplementationMixin<KeyType, Derived>::depth() const noexcept {
    return Derived::basis_traits::depth(static_cast<const Derived *>(this)->m_impl);
}
template <typename KeyType, typename Derived>
deg_t WordLikeBasisImplementationMixin<KeyType, Derived>::degree(const KeyType& key) const noexcept {
    return Derived::basis_traits::degree(
        static_cast<const Derived*>(this)->m_impl,
        key
        );
}
template <typename KeyType, typename Derived>
deg_t WordLikeBasisImplementationMixin<KeyType, Derived>::size(deg_t degree) const noexcept {
    return Derived::basis_traits::size(
        static_cast<const Derived*>(this)->m_impl,
        degree
        );
}
template <typename KeyType, typename Derived>
let_t WordLikeBasisImplementationMixin<KeyType, Derived>::first_letter(const KeyType &key) const noexcept {
    return Derived::basis_traits::first_letter(
        static_cast<const Derived*>(this)->m_impl,
        key
        );
}
template <typename KeyType, typename Derived>
dimn_t WordLikeBasisImplementationMixin<KeyType, Derived>::start_of_degree(deg_t degree) const noexcept {
    return Derived::basis_traits::start_of_degree(
        static_cast<const Derived*>(this)->m_impl,
        degree
        );
}
template <typename KeyType, typename Derived>
pair<optional<KeyType>, optional<KeyType>> WordLikeBasisImplementationMixin<KeyType, Derived>::parents(const KeyType &key) const {
    return Derived::basis_traits::parents(
        static_cast<const Derived*>(this)->m_impl,
        key
        );
}
template <typename KeyType, typename Derived>
key_type WordLikeBasisImplementationMixin<KeyType, Derived>::key_of_letter(let_t letter) const noexcept {
    return Derived::basis_traits::key_of_letter(
        static_cast<const Derived*>(this)->m_impl,
        letter
        );
}
template <typename KeyType, typename Derived>
bool WordLikeBasisImplementationMixin<KeyType, Derived>::letter(const KeyType &key) const {
    return Derived::basis_traits::letter(
        static_cast<const Derived*>(this)->m_impl,
        key
        );
}

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_IMPL_H_
