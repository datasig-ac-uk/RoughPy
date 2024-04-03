#ifndef UINT8_TYPE_H_
#define UINT8_TYPE_H_
#include "devices/fundamental_type.h"
#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
namespace rpy {
namespace devices {
extern template class RPY_LOCAL FundamentalType<uint8_t>;
extern RPY_LOCAL const FundamentalType<uint8_t> uint8_type;
}
}
#endif // UINT8_TYPE_H_