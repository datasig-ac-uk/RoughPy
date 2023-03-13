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
    static auto key(const Iter& it) noexcept -> decltype(it->first) {
        return it->first;
    }
    static auto value(const Iter& it) noexcept ->decltype(it->second) {
        return it->second;
    }
};

template <typename RealBasis, typename Iter>
class AlgebraIteratorImplementation : public AlgebraIteratorInterface {
    Iter m_iter;
    const RealBasis* p_basis;

    using itraits = iterator_helper_trait<Iter>;
public:

    AlgebraIteratorImplementation(Iter iter, const RealBasis* basis)
        : AlgebraIteratorInterface(Basis(p_basis)), m_iter(std::move(iter)), p_basis(basis)
    {}

    key_type key() const noexcept override {
        return basis_info<RealBasis>::convert_key(*p_basis, itraits::key(m_iter));
    }
    scalars::Scalar value() const noexcept override {
        using trait = scalars::scalar_type_trait<decltype(itraits::value(m_iter))>;
        return trait::make(itraits::value(m_iter));
    }
    std::shared_ptr<AlgebraIteratorInterface> clone() const override {
        return std::shared_ptr<AlgebraIteratorInterface>(new AlgebraIteratorImplementation(m_iter, p_basis));
    }
    void advance() override {
        ++m_iter;
    }
    bool equals(const AlgebraIteratorInterface &other) const noexcept override {
        return m_iter == static_cast<const AlgebraIteratorImplementation&>(other).m_iter;
    }
};


}
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_ITERATOR_IMPL_H_
