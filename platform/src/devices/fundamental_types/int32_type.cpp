
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

}// namespace dtl

template class FundamentalType<int32_t>;

}// namespace devices
}// namespace rpy
template <>
TypePtr devices::get_type<int32_t>()
{
    return FundamentalType<int32_t>::get();
}
