#include "uint16_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<uint16_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<uint16_t>
    devices::uint16_type("u16", "uint16");