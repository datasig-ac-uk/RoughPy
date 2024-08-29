
#include "algorithm_impl.h"
#include "devices/algorithms.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct ShiftLeftFunctor {
    void operator()(Buffer& buffer) const
    {
        auto slice = buffer.as_mut_slice<T>();
        auto dst = slice.end();
        auto begin = slice.begin();
        auto src = dst - 1;
        ;

        while (src != begin) { *(--dst) = std::move(*(--src)); }

        for (; dst != begin; --dst) { construct_inplace(std::addressof(dst)); }
    }
};
}// namespace

void AlgorithmDrivers::shift_left(Buffer& buffer) const
{
    auto host_mapped = buffer.map();
    algorithms::do_algorithm<void, ShiftLeftFunctor>(host_mapped);
}
