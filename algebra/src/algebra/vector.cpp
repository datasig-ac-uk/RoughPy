//
// Created by sam on 1/29/24.
//

#include "vector.h"
#include "roughpy/core/container/vector.h"

#include "basis.h"
#include "basis_key.h"
#include "key_algorithms.h"
#include "vector_iterator.h"

#include <roughpy/core/ranges.h>
#include <roughpy/devices/core.h>
#include <roughpy/devices/algorithms.h>

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <utility>

using namespace rpy;
using namespace algebra;

std::ostream& algebra::operator<<(std::ostream& os, const Vector& value)
{
    const auto basis = value.basis();
    os << '{';
    for (const auto& item : value) {
        os << ' ' << item->second << '(' << basis->to_string(item->first)
           << ')';
    }
    return os << " }";
}
