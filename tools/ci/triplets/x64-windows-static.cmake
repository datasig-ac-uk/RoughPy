include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE static)

if (PORT MATCHES "gmp")
    set(VCPKG_LIBRARY_LINKAGE dynamic)
endif ()
