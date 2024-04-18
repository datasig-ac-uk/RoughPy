
#include "uint64_type.h"

using namespace rpy;
using namespace rpy::devices;


namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<uint64_t> {
    static constexpr string_view id = "u64";
    static constexpr string_view name = "uint64";
};

}

template class FundamentalType<uint64_t>;

}
}

