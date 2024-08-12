//
// Created by sam on 09/08/24.
//

#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <algorithm>
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {
template <typename T>
struct DefaultFillFunctor {

    void operator()(Buffer& buffer) const
    {
        auto slice = buffer.as_mut_slice<T>();
        std::uninitialized_default_construct_n(slice.data(), slice.size());
    }
};
}// namespace

void AlgorithmDrivers::default_fill(Buffer& buffer) const
{
    auto dst_host_mapped = buffer.map();
    algorithms::do_algorithm<void, DefaultFillFunctor>(dst_host_mapped);
}