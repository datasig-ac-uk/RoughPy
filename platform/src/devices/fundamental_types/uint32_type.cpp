
#include "uint32_type.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<uint32_t> {
    static constexpr string_view id = "u32";
    static constexpr string_view name = "uint32";
};

}// namespace dtl

template class FundamentalType<uint32_t>;

}// namespace devices
}// namespace rpy

template <>
TypePtr devices::get_type<uint32_t>()
{
    return FundamentalType<uint32_t>::get();
}
