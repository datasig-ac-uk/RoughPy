
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

}

template class FundamentalType<int16_t>;

}
}

