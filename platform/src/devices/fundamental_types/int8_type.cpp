
#include "int8_type.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<int8_t> {
    static constexpr string_view id = "i8";
    static constexpr string_view name = "int8";
};

}// namespace dtl

template class FundamentalType<int8_t>;

}// namespace devices
}// namespace rpy
