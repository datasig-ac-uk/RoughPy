

include(FindPackageHandleStandardArgs)

set(_python_bin_dir "")
if(DEFINED Python_EXECUTABLE)
    cmake_path(GET Python_EXECUTABLE PARENT_PATH _python_bin_dir)
endif()

find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build
        DOC "Path to shpinx-build executable"
        HINTS ${_python_bin_dir}
)


find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_EXECUTABLE)

