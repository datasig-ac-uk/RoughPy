
#include "int64_type.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<int64_t> {
    static constexpr string_view id = "i64";
    static constexpr string_view name = "int64";
};

}// namespace dtl

template class FundamentalType<int64_t>;

}// namespace devices
}// namespace rpy
template <>
TypePtr devices::get_type<int64_t>()
{
    return FundamentalType<int64_t>::get();
}
