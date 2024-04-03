#ifndef UINT32_TYPE_H_
#define UINT32_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<uint32_t>;
extern RPY_LOCAL const FundamentalType<uint32_t> uint32_type;
}
}
#endif // UINT32_TYPE_H_