
#include "int32_type.h"

using namespace rpy;
using namespace rpy::devices;


namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<int32_t> {
    static constexpr string_view id = "i32";
    static constexpr string_view name = "int32";
};

}

template class FundamentalType<int32_t>;



template <>
const Type* devices::get_type<int32_t>() { return FundamentalType<int32_t>::get(); }

}
}
