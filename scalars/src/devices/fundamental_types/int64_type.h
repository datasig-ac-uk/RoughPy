#ifndef INT64_TYPE_H_
#define INT64_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<int64_t>;
extern RPY_LOCAL const FundamentalType<int64_t> int64_type;
}
}
#endif // INT64_TYPE_H_