
#include "int16_type.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<int16_t> {
    static constexpr string_view id = "i16";
    static constexpr string_view name = "int16";
};

}// namespace dtl

template class FundamentalType<int16_t>;

}// namespace devices
}// namespace rpy
template <>
TypePtr devices::get_type<int16_t>()
{
    return FundamentalType<int16_t>::get();
}
