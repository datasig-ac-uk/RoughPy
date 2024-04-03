#ifndef INT8_TYPE_H_
#define INT8_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<int8_t>;
extern RPY_LOCAL const FundamentalType<int8_t> int8_type;
}
}
#endif // INT8_TYPE_H_