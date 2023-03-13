//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H

#include "algebra_fwd.h"

#include <libalgebra_lite/dense_vector.h>
#include <libalgebra_lite/free_tensor.h>
#include <libalgebra_lite/lie.h>
#include <libalgebra_lite/shuffle_tensor.h>
#include <libalgebra_lite/sparse_vector.h>
#include <libalgebra_lite/vector.h>

namespace rpy {
namespace algebra {
namespace dtl {

template <typename VT>
using storage_type = lal::dtl::standard_storage<VT>;

template <VectorType>
struct vector_type_selector;

template <>
struct vector_type_selector<VectorType::Dense> {

    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::dense_vector, storage_type>;

    template <typename C>
    using shuffle_tensor = lal::shuffle_tensor<C, lal::dense_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::dense_vector, storage_type>;
};

template <>
struct vector_type_selector<VectorType::Sparse> {

    template <typename C>
    using free_tensor = lal::free_tensor<C, lal::sparse_vector, storage_type>;

    template <typename C>
    using shuffle_tensor = lal::shuffle_tensor<C, lal::sparse_vector, storage_type>;

    template <typename C>
    using lie = lal::lie<C, lal::sparse_vector, storage_type>;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_LITE_VECTOR_SELECTOR_H
