#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H_
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H_

#include "algebra_fwd.h"

#include "basis.h"

namespace rpy {
namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT TensorBasisInterface : public BasisInterface<>
{
public:
    ~TensorBasisInterface() override;
};


extern template class ROUGHPY_ALGEBRA_EXPORT Basis<TensorBasisInterface, OrderedBasisInterface<>, WordLikeBasisInterface<>>;

using TensorBasis = Basis<TensorBasisInterface,
                          OrderedBasisInterface<>,
                          WordLikeBasisInterface<> >;

}
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H_
