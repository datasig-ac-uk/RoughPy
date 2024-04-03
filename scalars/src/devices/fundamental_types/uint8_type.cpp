#include "uint8_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<uint8_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<uint8_t>
    devices::uint8_type("u8", "uint8");