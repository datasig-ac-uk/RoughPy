set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# HAVE_FPU is on for all native platforms, but not by default when building in a
# Linux container on MacOS (e.g. in devcontainer), so it is set explicitly here.
if (PORT MATCHES "mpg123")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS "-DHAVE_FPU=1")
endif()
