
#include "devices/algorithms.h"

#include "devices/algorithms/algorithm_impl.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

template <typename S, typename T>
struct CopyFunctor {
    void operator()(Buffer& dst, const Buffer& src) const
    {
        auto dst_slice = dst.as_mut_slice<S>();
        const auto src_slice = src.as_slice<T>();

        rpy::ranges::copy(src_slice, dst_slice.data());
    }
};

void AlgorithmDrivers::copy(Buffer& dest, const Buffer& source) const
{
    auto dest_host_mapped = dest.map();
    const auto source_host_view = source.map();
    algorithms::do_algorithm<void, CopyFunctor>(
            dest_host_mapped,
            source_host_view
    );
}
