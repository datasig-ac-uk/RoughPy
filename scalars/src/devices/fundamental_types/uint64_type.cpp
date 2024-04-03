#include "uint64_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<uint64_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<uint64_t>
    devices::uint64_type("u64", "uint64");