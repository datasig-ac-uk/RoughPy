#ifndef UINT64_TYPE_H_
#define UINT64_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<uint64_t>;
extern RPY_LOCAL const FundamentalType<uint64_t> uint64_type;
}
}
#endif // UINT64_TYPE_H_