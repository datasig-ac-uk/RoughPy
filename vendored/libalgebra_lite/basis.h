//
// Created by user on 07/02/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_BASIS_H
#define LIBALGEBRA_LITE_INCLUDE_BASIS_H

#include "implementation_types.h"

#include <memory>

#include "detail/notnull.h"

namespace lal {



template <typename Basis>
using basis_pointer = dtl::not_null<const Basis>;

} // namespace lal

#endif //LIBALGEBRA_LITE_INCLUDE_BASIS_H
