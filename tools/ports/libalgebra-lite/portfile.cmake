
vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REP datasig-ac-uk/libalgebra-lite
        REF main
)


vcpkg_cmake_configure(
        SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
