//
// Created by sam on 16/02/24.
//

#include "tensor_word.h"



using namespace rpy;
using namespace rpy::algebra;

string_view TensorWord::key_type() const noexcept
{
    return s_basis_type;
}
BasisPointer TensorWord::basis() const noexcept
{
    return BasisKeyInterface::basis();
}
