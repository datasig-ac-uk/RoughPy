#ifndef UINT16_TYPE_H_
#define UINT16_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<uint16_t>;
extern RPY_LOCAL const FundamentalType<uint16_t> uint16_type;
}
}
#endif // UINT16_TYPE_H_