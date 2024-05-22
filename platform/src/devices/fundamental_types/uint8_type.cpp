
#include "uint8_type.h"

using namespace rpy;
using namespace rpy::devices;


namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<uint8_t> {
    static constexpr string_view id = "u8";
    static constexpr string_view name = "uint8";
};

}

template class FundamentalType<uint8_t>;

template <>
const Type* devices::get_type<uint8_t>() { return FundamentalType<uint8_t>::get(); }

}
}
