
#include "algorithm_impl.h"
#include "devices/algorithms.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct ShiftRightFunctor {
    void operator()(Buffer& buffer) const
    {
        auto slice = buffer.as_mut_slice<T>();
        auto src = slice.begin();
        auto dst = src + 1;
        auto end = slice.end();
        for (; dst != end; ++src, ++dst) { *dst = std::move(*src); }
        for (dst = slice.begin(); dst != slice.begin() + 1; ++dst) {
            construct_inplace(std::addressof(dst));
        }
    }
};

}// namespace

void AlgorithmDrivers::shift_right(Buffer& buffer) const
{
    auto host_mapped = buffer.map();
    algorithms::do_algorithm<void, ShiftRightFunctor>(host_mapped);
}
