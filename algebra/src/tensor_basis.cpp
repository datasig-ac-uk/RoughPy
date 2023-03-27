//
// Created by user on 27/03/23.
//

#include "tensor_basis.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy {
namespace algebra {
template class Basis<TensorBasisInterface,
                     OrderedBasisInterface<>,
                     WordLikeBasisInterface<>>;
}
}// namespace rpy

TensorBasisInterface::~TensorBasisInterface() = default;
