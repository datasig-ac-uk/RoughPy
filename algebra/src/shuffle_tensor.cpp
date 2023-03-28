//
// Created by user on 06/03/23.
//

#include "shuffle_tensor.h"
#include "context.h"


namespace rpy { namespace algebra {

template class AlgebraInterface<ShuffleTensor, TensorBasis>;
template class AlgebraBase<ShuffleTensorInterface>;

}}
