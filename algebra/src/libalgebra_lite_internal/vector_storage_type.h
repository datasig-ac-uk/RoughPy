//
// Created by user on 18/07/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_VECTOR_STORAGE_TYPE_H_
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_VECTOR_STORAGE_TYPE_H_

#include <libalgebra_lite/dense_vector.h>
#include <libalgebra_lite/sparse_vector.h>

#include <roughpy/algebra/algebra_fwd.h>

namespace rpy {
namespace algebra {
namespace dtl {

template <template <typename, typename, typename...> class VT>
struct vector_storage_type;


template <>
struct vector_storage_type<lal::dense_vector> {
    static constexpr VectorType value = VectorType::Dense;
};

template <>
struct vector_storage_type<lal::sparse_vector> {
    static constexpr VectorType value = VectorType::Sparse;
};


}
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_VECTOR_STORAGE_TYPE_H_
