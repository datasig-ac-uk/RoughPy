//
// Created by sam on 3/30/24.
//

#include "double_type.h"

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<double> {
    static constexpr string_view id = "f64";
    static constexpr string_view name = "f64";
};

}// namespace dtl
template class FundamentalType<double>;

}// namespace devices
}// namespace rpy
template <>
TypePtr rpy::devices::get_type<double>()
{
    return FundamentalType<double>::get();
}
