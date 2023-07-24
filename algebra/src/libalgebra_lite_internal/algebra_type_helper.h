//
// Created by user on 18/07/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_HELPER_H_
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_HELPER_H_

#include <libalgebra_lite/free_tensor.h>
#include <libalgebra_lite/lie.h>
#include <libalgebra_lite/shuffle_tensor.h>

#include "vector_type_helper.h"
#include "vector_storage_type.h"

namespace rpy {
namespace algebra {
namespace dtl {

template <typename Algebra>
struct alg_details_of;

template <
        typename Coefficients, template <typename, typename> class
        LalVectorType,
        template <typename> class StorageModel>
struct alg_details_of<
        lal::free_tensor<Coefficients, LalVectorType, StorageModel>> {
    using type = lal::free_tensor<Coefficients, LalVectorType, StorageModel>;
    static constexpr AlgebraType alg_type = AlgebraType::FreeTensor;
    using interface_type = FreeTensorInterface;
    using wrapper_type = FreeTensor;
    using implementation_type
            = FreeTensorImplementation<type, OwnedStorageModel>;
};
template <
        typename Coefficients, template <typename, typename> class
        LalVectorType,
        template <typename> class StorageModel>
struct alg_details_of<
        lal::shuffle_tensor<Coefficients, LalVectorType, StorageModel>> {
    using type = lal::shuffle_tensor<Coefficients, LalVectorType, StorageModel>;
    static constexpr AlgebraType alg_type = AlgebraType::ShuffleTensor;
    using interface_type = ShuffleTensorInterface;
    using wrapper_type = ShuffleTensor;
    using implementation_type
            = AlgebraImplementation<interface_type, type, OwnedStorageModel>;
};

template <
        typename Coefficients, template <typename, typename> class VectorType,
        template <typename> class StorageModel>
struct alg_details_of<lal::lie<Coefficients, VectorType, StorageModel>> {
    using type = lal::lie<Coefficients, VectorType, StorageModel>;
    static constexpr AlgebraType alg_type = AlgebraType::Lie;
    using interface_type = LieInterface;
    using wrapper_type = Lie;
    using implementation_type
            = AlgebraImplementation<interface_type, type, OwnedStorageModel>;
};
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_ALGEBRA_TYPE_HELPER_H_
