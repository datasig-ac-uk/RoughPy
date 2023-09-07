//
// Created by sam on 01/09/23.
//

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/device/device_handle.h>


#include <boost/smart_ptr/intrusive_ptr.hpp>

#include "opencl/open_cl_device.h"

#include "opencl/ocl_buffer.h"
#include "opencl/ocl_event.h"
#include "opencl/ocl_kernel.h"
#include "opencl/ocl_queue.h"



using namespace rpy;
using namespace rpy::device;

namespace rpy { namespace device {

boost::intrusive_ptr<DeviceHandle> get_device(DeviceInfo info);

}}


static const OCLKernelInterface s_ocl_kernel_iface;

boost::intrusive_ptr<DeviceHandle> device::get_device(DeviceInfo info) {





}
