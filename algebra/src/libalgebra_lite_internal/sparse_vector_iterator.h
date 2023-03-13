//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SPARSE_VECTOR_ITERATOR_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SPARSE_VECTOR_ITERATOR_H

#include "algebra_iterator.h"
#include "algebra_iterator_impl.h"

#include <libalgebra_lite/sparse_vector.h>

namespace rpy { namespace algebra {

template <typename MapType, typename Iterator>
struct iterator_helper_trait<lal::dtl::sparse_iterator<MapType, Iterator>> {
    using iter_t = lal::dtl::sparse_iterator<MapType, Iterator>;

    static auto key(const iter_t& it) noexcept -> decltype(it->key()) {
        return it->key();
    }
    static auto value(const iter_t& it) noexcept -> decltype(it->value()) {
        return it->value();
    }
};

}}

#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_SPARSE_VECTOR_ITERATOR_H
