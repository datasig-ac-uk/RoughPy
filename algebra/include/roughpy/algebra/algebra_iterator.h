#ifndef ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_

#include "algebra_fwd.h"
#include "roughpy_algebra_export.h"
#include "basis.h"

#include <roughpy/scalars/scalar.h>

#include <memory>

namespace rpy {
namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT AlgebraIteratorInterface {
    Basis m_basis;

protected:
    explicit AlgebraIteratorInterface(Basis basis) : m_basis(std::move(basis))
    {}

public:
    virtual ~AlgebraIteratorInterface() = default;

    const Basis& basis() const { return m_basis; };
    virtual key_type key() const noexcept = 0;
    virtual scalars::Scalar value() const noexcept = 0;

    virtual std::shared_ptr<AlgebraIteratorInterface> clone() const = 0;
    virtual void advance() = 0;
    virtual bool equals(const AlgebraIteratorInterface& other) const noexcept = 0;

};


class ROUGHPY_ALGEBRA_EXPORT AlgebraIteratorItem {
    std::shared_ptr<AlgebraIteratorInterface> p_interface;

    friend class AlgebraIterator;

    AlgebraIteratorItem(std::shared_ptr<AlgebraIteratorInterface> interface)
        : p_interface(std::move(interface))
    {}


public:
    const Basis& basis() const;
    key_type key() const noexcept;
    scalars::Scalar value() const noexcept;

    AlgebraIteratorItem* operator->() noexcept { return this; }


};


class ROUGHPY_ALGEBRA_EXPORT AlgebraIterator {
    std::shared_ptr<AlgebraIteratorInterface> p_interface;
    std::uintptr_t m_tag;

public:

    using value_type = AlgebraIteratorItem;
    using reference = AlgebraIteratorItem;
    using pointer = AlgebraIteratorItem;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    AlgebraIterator(std::shared_ptr<AlgebraIteratorInterface> interface,
                    std::uintptr_t tag)
        : p_interface(std::move(interface)), m_tag(tag) {}

    AlgebraIterator() = default;
    AlgebraIterator(const AlgebraIterator& arg);
    AlgebraIterator(AlgebraIterator&& arg) noexcept;

    AlgebraIterator& operator=(const AlgebraIterator& arg);
    AlgebraIterator& operator=(AlgebraIterator&& arg) noexcept;



    AlgebraIterator& operator++();
    const AlgebraIterator operator++(int);
    AlgebraIteratorItem operator*() const;
    AlgebraIteratorItem operator->() const;

    bool operator==(const AlgebraIterator& other) const;
    bool operator!=(const AlgebraIterator& other) const;

};

}
}
#endif // ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_H_
