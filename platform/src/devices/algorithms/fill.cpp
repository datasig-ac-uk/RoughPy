
#include "devices/algorithms.h"
#include "algorithm_impl.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct FillFunctor
{
    void operator()(Buffer& buffer, ConstReference value) const {
        rpy::ranges::fill(buffer.as_mut_slice<T>(), value.value<T>());
    }
};

}


void AlgorithmDrivers::fill(Buffer& dst, ConstReference value) const
{
    auto dst_host_mapped = dst.map();
    algorithms::do_algorithm<void, FillFunctor>(dst_host_mapped, value);
}
