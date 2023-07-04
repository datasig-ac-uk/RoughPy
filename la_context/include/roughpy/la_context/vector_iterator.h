#ifndef ROUGHPY_LA_CONTEXT_VECTOR_ITERATOR_H_
#define ROUGHPY_LA_CONTEXT_VECTOR_ITERATOR_H_

#include <roughpy/algebra/algebra_iterator.h>

#include <libalgebra/iterators.h>

namespace rpy {
namespace algebra {

template <typename Item>
struct iterator_helper_trait<alg::vectors::iterators::vector_iterator<Item>> {
    using iter_t = alg::vectors::iterators::vector_iterator<Item>;

    static auto key(const iter_t& it) noexcept -> decltype(it->key())
    {
        return it->key();
    }

    static auto value(const iter_t& it) noexcept -> decltype(it->value())
    {
        return it->value();
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_LA_CONTEXT_VECTOR_ITERATOR_H_
