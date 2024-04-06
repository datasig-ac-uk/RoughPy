
#include "algorithm_impl.h"
#include "devices/algorithms.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;


namespace {

template <typename T>
struct ReverseFunctor
{
    void operator()(Buffer& buffer) const
    {
        rpy::ranges::reverse(buffer.as_mut_slice<T>());
    }
};

}


void AlgorithmDrivers::reverse(Buffer& buffer) const
{
    auto host_mapped = buffer.map();
    algorithms::do_algorithm<void, ReverseFunctor>(host_mapped);
}
