//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H

#include "algebra_iterator.h"
#include "algebra_iterator_impl.h"

#include <libalgebra_lite/dense_vector.h>

namespace rpy { namespace algebra {

template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_helper_trait<lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>> {
    using iter_t = lal::dtl::dense_vector_iterator<Basis, Coefficients, Iterator>;

    static auto key(iter_t &it) noexcept -> decltype(it->key()) {
        return it->key();
    }
    static auto value(iter_t &it) noexcept -> decltype(it->value()) {
        return it->value();
    }
};

template <typename Basis, typename Coefficients, typename Iterator>
struct iterator_helper_trait<lal::dtl::dense_vector_const_iterator<Basis, Coefficients, Iterator>> {
    using iter_t = lal::dtl::dense_vector_const_iterator<Basis, Coefficients, Iterator>;

    static auto key(iter_t &it) noexcept -> decltype(it->key()) {
        return it->key();
    }
    static auto value(iter_t &it) noexcept -> decltype(it->value()) {
        return it->value();
    }
};


}}


#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_DENSE_VECTOR_ITERATOR_H
