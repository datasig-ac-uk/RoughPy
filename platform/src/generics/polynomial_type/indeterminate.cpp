//
// Created by sam on 26/11/24.
//

#include "indeterminate.h"

#include <ostream>

using namespace rpy;
using namespace rpy::generics;


std::ostream& generics::operator<<(std::ostream& os, const Indeterminate& value) {
    return os << value.prefix() << value.index();
}