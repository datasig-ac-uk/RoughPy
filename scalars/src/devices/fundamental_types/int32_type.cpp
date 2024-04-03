#include "int32_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<int32_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<int32_t>
    devices::int32_type("i32", "int32");