
vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REPO datasig-ac-uk/libalgebra-lite
        REF 53b98d00bd4bff96247ac0bf881b27dfa0e0ad5b
        SHA512 3e3bb26533a2213a6ee87e43c006a57c9a65a00f5851b6f20496603d03fc2e35463aa3c009b151d78034e8670936db27921bb74601825fc9c404f476090012ac
        HEAD_REF main
)

vcpkg_cmake_configure(
        SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME "libalgebra-lite")


file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(INSTALL "${SOURCE_PATH}/LICENSE"
        DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
        RENAME copyright)

#configure_file("${CMAKE_CURRENT_LIST_DIR}/usage" "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage" COPYONLY)
