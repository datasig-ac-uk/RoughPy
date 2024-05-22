
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

}// namespace dtl

template class FundamentalType<uint64_t>;

template <>
const Type* devices::get_type<uint64_t>()
{
    return FundamentalType<uint64_t>::get();
}

}// namespace devices
}// namespace rpy
