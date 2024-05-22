//
// Created by sam on 3/30/24.
//

#include "float_type.h"

using namespace rpy;
using namespace rpy::devices;
namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<float> {
    static constexpr string_view id = "f32";
    static constexpr string_view name = "f32";
};

}// namespace dtl
template class FundamentalType<float>;

template <>
const Type* rpy::devices::get_type<float>()
{
    return FundamentalType<float>::get();
}
}// namespace devices
}// namespace rpy
