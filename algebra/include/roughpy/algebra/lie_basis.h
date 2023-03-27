#ifndef ROUGHPY_ALGEBRA_LIE_BASIS_H_
#define ROUGHPY_ALGEBRA_LIE_BASIS_H_

#include "basis.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT LieBasisInterface : public BasisInterface<> {
};

extern template class ROUGHPY_ALGEBRA_EXPORT Basis<LieBasisInterface, OrderedBasisInterface<>, WordLikeBasisInterface<>>;


using LieBasis = Basis<LieBasisInterface, OrderedBasisInterface<>, WordLikeBasisInterface<>>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_LIE_BASIS_H_
