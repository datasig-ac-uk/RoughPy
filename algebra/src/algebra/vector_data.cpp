//
// Created by sam on 22/03/24.
//

#include "vector.h"

using namespace rpy;
using namespace rpy::algebra;

VectorData::~VectorData() = default;

Rc<VectorData> VectorData::empty_like() const noexcept
{
    return make_default(basis());
}

bool VectorData::is_sparse() const noexcept { return false; }
bool VectorData::is_contiguous() const noexcept { return true; }

Rc<VectorData> VectorData::make_default(BasisPointer basis) noexcept
{

    return nullptr;
}
