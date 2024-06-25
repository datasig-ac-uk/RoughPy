//
// Created by sam on 3/18/24.
//

#include "tensor_product_basis_key.h"

using namespace rpy;
using namespace rpy::algebra;

string_view TensorProductBasisKey::key_type() const noexcept
{
    return key_name;
}
