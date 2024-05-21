//
// Created by sam on 3/30/24.
//

#include "double_type.h"


namespace rpy { namespace devices {

namespace dtl {
template <>
struct IDAndNameOfFType<double> {
    static constexpr string_view id = "f64";
    static constexpr string_view name = "f64";
};

}
template class FundamentalType<double>;

}}
