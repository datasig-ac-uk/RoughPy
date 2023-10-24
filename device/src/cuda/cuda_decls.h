//
// Created by sam on 24/10/23.
//

#ifndef ROUGHPY_CUDA_DECLS_H
#define ROUGHPY_CUDA_DECLS_H

#include <roughpy/device/core.h>

namespace rpy {
namespace devices {

class CUDABuffer;
class CUDADeviceHandle;
class CUDAEvent;
class CUDAKernel;
class CUDAQueue;


using CUDADevice = boost::intrusive_ptr<const CUDADeviceHandle>;

}
}// namespace rpy

#endif// ROUGHPY_CUDA_DECLS_H
