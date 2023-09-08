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

#include <boost/functional/hash.hpp>

#include <mutex>
#include <unordered_map>

using namespace rpy;
using namespace rpy::device;


static std::mutex s_lock;
static std::unordered_map<
        DeviceInfo, boost::intrusive_ptr<DeviceHandle>, boost::hash<DeviceInfo>>
        s_device_cache;

static constexpr cl_uint s_max_num_platforms = 16;
static constexpr cl_uint s_max_num_devices = 16;

boost::intrusive_ptr<DeviceHandle> device::get_device(DeviceInfo info) {
    std::lock_guard<std::mutex> access(s_lock);

    auto& entry = s_device_cache[info];
    if (!entry) {

        cl_device_type search_type;
        switch (info.device_type) {
            case CPU:
                search_type = CL_DEVICE_TYPE_CPU;
                break;
            case CUDA:
            case CUDAHost:
            case CUDAManaged:
            case Metal:
            case ROCM:
            case ROCMHost:
            case Vulkan:
            case WebGPU:
                search_type = CL_DEVICE_TYPE_GPU;
                break;
            case ExtDev:
                RPY_THROW(std::runtime_error, "ExtDev is not supported");
            case VPI: RPY_FALLTHROUGH;
            case Hexagon:
                search_type = CL_DEVICE_TYPE_ACCELERATOR;
                break;
            case OpenCL: RPY_FALLTHROUGH;
            case OneAPI:
                search_type = CL_DEVICE_TYPE_DEFAULT;
        }


        cl_int rc;
        cl_uint num_platforms = 0;
        std::vector<cl_platform_id> plats(s_max_num_platforms);
        rc = clGetPlatformIDs(s_max_num_platforms, plats.data(), &num_platforms);

        RPY_CHECK(rc == CL_SUCCESS);

        std::vector<cl_device_id> devices(s_max_num_devices);
        std::vector<cl_device_id> candidates;
        candidates.reserve(s_max_num_devices);
        for (cl_uint i=0; i<num_platforms; ++i) {
            auto id = plats[i];
            cl_uint num_devices;
            rc = clGetDeviceIDs(id, search_type, s_max_num_devices,
                                devices.data(), &num_devices);
            if (rc != CL_SUCCESS || num_devices == 0) {
                continue;
            }

            for (cl_uint dev_idx=0; dev_idx < num_devices; ++dev_idx) {
                const auto& dev = devices[dev_idx];
                cl_bool is_available;
                rc = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE,
                                     sizeof(is_available),
                                     &is_available, nullptr);
                if (rc != CL_SUCCESS || !is_available) { continue; }

                // Other checks?
                candidates.push_back(dev);
            }
        }

        if (candidates.empty()) {
            RPY_THROW(std::runtime_error, "could not find appropriate device");
        }

        entry = new OpenCLDevice(candidates[0]);
    }
    return entry;
}
