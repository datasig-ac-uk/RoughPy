#ifndef ROUGHPY_DEVICE_CORE_H_
#define ROUGHPY_DEVICE_CORE_H_

#ifdef __NVCC__
#  include <cuda.h>

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE __inline__

#elif defined(__HIPCC__)

#  define RPY_DEVICE __device__
#  define RPY_HOST __host__
#  define RPY_DEVICE_HOST __device__ __host__
#  define RPY_KERNEL __global__
#  define RPY_DEVICE_SHARED __shared__
#  define RPY_STRONG_INLINE

#else
#  define RPY_DEVICE
#  define RPY_HOST
#  define RPY_DEVICE_HOST
#  define RPY_KERNEL
#  define RPY_DEVICE_SHARED
#  define RPY_STRONG_INLINE

#endif

namespace rpy {
namespace device {

using dindex_t = int;
using dsize_t = unsigned int;

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_CORE_H_
