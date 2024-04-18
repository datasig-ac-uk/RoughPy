//
// Created by sam on 4/7/24.
//

#include "algorithm_drivers.h"
#include <boost/smart_ptr/intrusive_ptr.hpp>

using namespace rpy;
using namespace rpy::devices;

AlgorithmDriversPtr algorithms::get_builtin_algorithms() noexcept
{
    static AlgorithmDriversPtr algos(new AlgorithmDrivers);
    return algos;
}

AlgorithmDrivers::~AlgorithmDrivers() = default;
