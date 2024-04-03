#include "uint32_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<uint32_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<uint32_t>
    devices::uint32_type("u32", "uint32");