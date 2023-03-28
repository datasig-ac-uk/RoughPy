#ifndef ROUGHPY_ALGEBRA_LIE_BASIS_H_
#define ROUGHPY_ALGEBRA_LIE_BASIS_H_

#include "basis.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT LieBasisInterface
    : public make_basis_interface<LieBasisInterface, rpy::key_type,
                                  OrderedBasisInterface,
                                  WordLikeBasisInterface>
{
};

extern template class ROUGHPY_ALGEBRA_EXPORT Basis<LieBasisInterface>;


using LieBasis = Basis<LieBasisInterface>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LIE_BASIS_H_
