#include "devices/algorithms.h"
#include "algorithm_impl.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;


namespace {

template <typename T>
struct MinFunctor
{
    void operator()(const Buffer& buffer, Reference out) const
    {
        out.value<T>() = rpy::ranges::min(buffer.as_slice<T>());
    }
};

}



void AlgorithmDrivers::min(const Buffer& buffer, Reference out) const
{
    const auto host_view = buffer.map();
    algorithms::do_algorithm<void, MinFunctor>(host_view, std::move(out));
}
