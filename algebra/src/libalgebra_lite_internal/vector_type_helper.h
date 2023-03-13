//
// Created by user on 07/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_VECTOR_TYPE_HELPER_H
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_VECTOR_TYPE_HELPER_H

#include "algebra_fwd.h"

#include <libalgebra_lite/sparse_vector.h>
#include <libalgebra_lite/dense_vector.h>


namespace rpy { namespace algebra { namespace dtl {

template <template <typename, typename> class VT>
struct vector_type_helper;

template <>
struct vector_type_helper<lal::sparse_vector> {
    static constexpr VectorType vtype = VectorType::Sparse;
};

template <>
struct vector_type_helper<lal::dense_vector> {
    static constexpr VectorType vtype = VectorType::Dense;
};




}}}


#endif//ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_VECTOR_TYPE_HELPER_H
