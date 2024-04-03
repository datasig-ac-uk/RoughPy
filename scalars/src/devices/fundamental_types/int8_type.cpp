#include "int8_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<int8_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<int8_t>
    devices::int8_type("i8", "int8");