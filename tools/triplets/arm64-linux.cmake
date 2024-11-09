set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# FIXME for discussion in PR. I had to add this to get containers working in MacOS, as it seems the macro is not set correctly by default.
if (PORT MATCHES "mpg123")
    set(VCPKG_CMAKE_CONFIGURE_OPTIONS "-DHAVE_FPU=1")
endif()
