#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H_
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H_

#include "algebra_fwd.h"

#include "basis.h"

namespace rpy {
namespace algebra {


class ROUGHPY_ALGEBRA_EXPORT TensorBasisInterface
    : public make_basis_interface<TensorBasisInterface, rpy::key_type,
                                  OrderedBasisInterface,
                                  WordLikeBasisInterface>
{
public:
    ~TensorBasisInterface() override;
};


extern template class ROUGHPY_ALGEBRA_EXPORT Basis<TensorBasisInterface>;

using TensorBasis = Basis<TensorBasisInterface>;

}
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H_
