#include "int64_type.h"
namespace rpy {
namespace devices {
template class FundamentalType<int64_t>;
}
}
using namespace rpy;
using namespace rpy::devices;
const FundamentalType<int64_t>
    devices::int64_type("i64", "int64");