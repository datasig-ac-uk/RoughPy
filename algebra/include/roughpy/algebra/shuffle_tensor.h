#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_

#include "algebra_base.h"

#include "tensor_basis.h"

namespace rpy {
namespace algebra {

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraInterface<ShuffleTensor, TensorBasis>;

using ShuffleTensorInterface = AlgebraInterface<ShuffleTensor, TensorBasis>;

extern template class ROUGHPY_ALGEBRA_EXPORT AlgebraBase<ShuffleTensorInterface>;

class ShuffleTensor : public AlgebraBase<ShuffleTensorInterface>
{
    using base_t = AlgebraBase<ShuffleTensorInterface>;
public:

    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensor;

    using base_t::base_t;

};

}
}

#endif // ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_
