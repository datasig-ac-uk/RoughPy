
#include "uint16_type.h"

using namespace rpy;
using namespace rpy::devices;


namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<uint16_t> {
    static constexpr string_view id = "u16";
    static constexpr string_view name = "uint16";
};

}

template class FundamentalType<uint16_t>;

}
}

