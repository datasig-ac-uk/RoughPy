#include "int16_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<int16_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<int16_t>
    devices::int16_type("i16", "int16");