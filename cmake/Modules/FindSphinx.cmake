

include(FindPackageHandleStandardArgs)

find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build
        DOC "Path to shpinx-build executable")


find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_EXECUTABLE)

