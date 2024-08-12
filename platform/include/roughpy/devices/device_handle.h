// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_DEVICE_HANDLE_H_
#define ROUGHPY_DEVICE_DEVICE_HANDLE_H_

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/filesystem.h>

#include <mutex>

#include "core.h"

#include "buffer.h"
#include "event.h"
#include "kernel.h"
#include "queue.h"

namespace rpy {
namespace devices {

struct ExtensionSourceAndOptions {
    containers::Vec<string> sources;
    string compile_options;
    containers::Vec<pair<string, string>> header_name_and_source;
    string link_options;
};

class AlgorithmsDispatcher;

/**
 * @class DeviceHandle
 * @brief Represents a handle to a device
 *
 * This class provides an interface to interact with a device. It is used as a
 * base class for specific device handle implementations.
 */
class ROUGHPY_DEVICES_EXPORT DeviceHandle : public RcBase<DeviceHandle>
{
    mutable std::recursive_mutex m_lock;
    mutable containers::HashMap<string, Kernel> m_kernel_cache;

    std::unique_ptr<AlgorithmsDispatcher> p_algorithms;

protected:
    using lock_type = std::recursive_mutex;
    using guard_type = std::lock_guard<lock_type>;

    lock_type& get_lock() const noexcept { return m_lock; }

public:
    DeviceHandle();

    virtual ~DeviceHandle();

    /**
     * @brief Checks if the device is a host device
     *
     * This method is used to determine if the device is a host device.
     *
     * @return true if the device is a host device, false otherwise.
     */
    RPY_NO_DISCARD virtual bool is_host() const noexcept;
    /**
     * @brief Get the type of the device
     *
     * Returns the type of the device represented by this handle.
     *
     * @return The type of the device. The specific type is defined by the
     * DeviceType enumeration.
     * @see DeviceType
     */
    RPY_NO_DISCARD virtual DeviceType type() const noexcept;
    /**
     * @brief Returns the category of the device handle
     *
     * This method returns the category of the device handle. The category
     * represents the type of device the handle is associated with.
     *
     * @return The category of the device handle. It is of type DeviceCategory.
     */
    RPY_NO_DISCARD virtual DeviceCategory category() const noexcept;

    /**
     * @brief Get the information of the device
     *
     * This method returns the information of the device represented by the
     * DeviceHandle object.
     *
     * @return The DeviceInfo object containing the device information.
     * @note This method is marked as noexcept, hence it does not throw any
     * exceptions.
     */
    RPY_NO_DISCARD virtual DeviceInfo info() const noexcept;

    /**
     * @brief Returns the path to the runtime library associated with the device
     *
     * This method returns an optional path that represents the location of the
     * runtime library associated with the device. If no runtime library is
     * associated or if the information is not available, an empty optional is
     * returned.
     *
     * @return An optional fs::path object representing the path to the runtime
     * library, or an empty optional if no runtime library is associated or the
     * information is not available.
     *
     * @note The returned path may or may not be an actual file on the file
     * system. It is solely provided as a means to represent the location of the
     * runtime library. Use operations provided by the fs::path class to
     * manipulate and operate on the path if necessary.
     *
     * @see fs::path
     */
    RPY_NO_DISCARD virtual optional<fs::path> runtime_library() const noexcept;

    //    virtual void launch_kernel(const void* kernel,
    //                               const void* launch_config,
    //                               void** args
    //                               ) = 0;

    /**
     * @brief Allocates a buffer of the specified size with the given type
     *
     * This method allocates a buffer on the device with the specified size and
     * type. The buffer is returned as a `Buffer` object.
     *
     * @param type The type information for the buffer
     * @param count The number of elements in the buffer
     * @return A `Buffer` object representing the allocated buffer on the device
     */
    RPY_NO_DISCARD virtual Buffer alloc(const Type& type, dimn_t count) const;

    /**
     * @brief Free the underlying resources associated with a buffer
     * @param buf buffer to free
     *
     * This does not destruct elements of the buffer, it just deallocates
     * the memory resources. This is equivalent to libc free.
     */
    virtual void raw_free(Buffer& buf) const;

    /**
     * @brief Checks if the device handle has a compiler
     *
     * This method is used to determine if the device handle has a compiler.
     * It returns a boolean value indicating whether or not a compiler is
     * present. The method is constant and does not throw exceptions.
     *
     * @return True if the device handle has a compiler, false otherwise.
     */
    virtual bool has_compiler() const noexcept;

    /**
     * @brief Registers a kernel with the device handle
     *
     * This method registers a kernel with the device handle. The kernel is
     * added to the kernel cache if it does not already exist.
     *
     * @param kernel The kernel to register with the device handle.
     * @return A const reference to the registered kernel in the kernel cache.
     *
     * @see Kernel, DeviceHandle
     */
    virtual const Kernel& register_kernel(Kernel kernel) const;

    /**
     * @brief Retrieves a kernel with the specified name
     *
     * This method retrieves a kernel object with the specified name from the
     * kernel cache. If the kernel with the specified name is found in the
     * cache, it is returned. Otherwise, an empty optional is returned.
     *
     * @param name The name of the kernel to retrieve
     *
     * @return An optional object containing the kernel with the specified name
     * if found, or an empty optional if not found
     */
    RPY_NO_DISCARD virtual optional<Kernel> get_kernel(const string& name
    ) const noexcept;

    RPY_NO_DISCARD virtual optional<Kernel>
    compile_kernel_from_str(const ExtensionSourceAndOptions& args) const;

    virtual void compile_kernels_from_src(const ExtensionSourceAndOptions& args
    ) const;

    RPY_NO_DISCARD virtual Event new_event() const;
    RPY_NO_DISCARD virtual Queue new_queue() const;

    /**
     * @brief Get the default queue associated with the device handle.
     *
     * This method returns the default queue associated with the device handle.
     * The default queue is the primary queue used for data transfer and command
     * execution.
     *
     * @return The default queue associated with the device handle.
     */
    RPY_NO_DISCARD virtual Queue get_default_queue() const;

    /**
     * @brief Get the UUID of the device handle
     *
     * This method returns the Universally Unique Identifier (UUID) of the
     * device handle. The UUID is used to uniquely identify a device handle.
     *
     * @return An optional containing the UUID of the device handle. If the UUID
     * is not available, an empty optional is returned.
     *
     * @note The returned UUID is of type boost::uuids::uuid.
     *
     * @par Example:
     * @code{.cpp}
     * optional<boost::uuids::uuid> deviceId = deviceHandle.uuid();
     * if (deviceId) {
     *     // Do something with the UUID
     * } else {
     *     // Handle case when UUID is not available
     * }
     * @endcode
     */
    RPY_NO_DISCARD virtual optional<boost::uuids::uuid> uuid() const noexcept;
    /**
     * @brief Retrieves information about the PCI bus for the device handle
     *
     * This function returns an optional instance of PCIBusInfo, which
     * represents information about the PCI bus associated with the device
     * handle. If the information is available, it will be returned; otherwise,
     * an empty optional will be returned.
     *
     * @return An optional instance of PCIBusInfo that contains the information
     *         about the PCI bus, or an empty optional if the information is not
     *         available.
     *
     * @see PCIBusInfo
     *
     */
    RPY_NO_DISCARD virtual optional<PCIBusInfo> pci_bus_info() const noexcept;

    /**
     * @brief Checks if the device handle supports a specific type
     * @param info The type information to check
     * @return True if the device handle supports the type, false otherwise
     *
     * This method is used to determine if the device handle supports a specific
     * type based on the provided type information. The type information is
     * passed as a reference to a TypeInfo object.
     *
     * The method returns true if the device handle supports the type
     * represented by the type information, and false otherwise.
     *
     * The method is marked as const, indicating that it does not modify the
     * internal state of the device handle.
     */
    RPY_NO_DISCARD virtual bool supports_type(const Type& info) const noexcept;

    RPY_NO_DISCARD virtual Event
    from_host(Buffer& dst, const BufferInterface& src, Queue& queue) const;

    virtual Event
    to_host(Buffer& dst, const BufferInterface& src, Queue& queue) const;

protected:
    void
    check_type_compatibility(const Type* primary, const Type* secondary) const;

public:
    /**
     * @brief Returns the algorithm drivers for the given primary and secondary
     * types
     * @param primary_type The primary type to get algorithms for
     * @param secondary_type The secondary type to get algorithms for. Default
     * value is nullptr.
     * @param check_conversion Flag indicating whether to check for type
     * compatibility. Default value is false.
     * @return AlgorithmDriversPtr Pointer to the algorithm drivers
     *
     * This method returns the algorithm drivers for the given primary and
     * secondary types. If secondary_type is nullptr, it is set to primary_type.
     * If check_conversion is true, the method checks for type compatibility
     * between primary_type and secondary_type. If both primary_type and
     * secondary_type are arithmetic, the method returns the built-in
     * algorithms. If no standard algorithms are available for the given primary
     * and secondary types, the method throws a std::runtime_error exception.
     */
    RPY_NO_DISCARD const AlgorithmsDispatcher& algorithms(
            const Type* primary_type,
            const Type* secondary_type = nullptr,
            bool check_conversion = false
    ) const
    {
        return *p_algorithms;
    }

    template <template <typename...> class Implementor, typename... Ts>
    void register_algorithm_drivers() const;

    template <template <typename...> class Implementor, typename... Ts>
    void register_algorithm_drivers(dtl::TypePtrify<Ts>...) const;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_DEVICE_HANDLE_H_
