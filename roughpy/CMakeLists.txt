

setup_roughpy_component(PyModule)




###############################################################################
#                                   pybind11                                  #
###############################################################################
# Pybind11 might have been installed using pip, which is probably the
# best way to handle it. In this case, we can get the cmake module dir
# by calling the pybind11 script installed here. If this is set, add it
# to the prefix path before we start looking for pybind11 below
if(NOT DEFINED pybind11_ROOT)
    execute_process(COMMAND
            "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
            RESULT_VARIABLE _python_pybind11_dir_found
            OUTPUT_VARIABLE _python_pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if (_python_pybind11_dir_found EQUAL 0)
        message(STATUS "Adding Pybind11 cmake dir ${_python_pybind11_dir}")
        set(pybind11_ROOT "${_python_pybind11_dir}" CACHE INTERNAL "")
        list(APPEND CMAKE_PREFIX_PATH "${_python_pybind11_dir}")
    endif ()
endif ()


# Pybind11 can either use it's own internal algorithm for finding Python, or
# use the FindPython module provided by CMake. We always want it to use the
# latter so we don't end up with two versions of Python out there.
set(PYBIND11_FINDPYTHON ON)

# We are not using vcpkg to fetch pybind11 because it "depends" on Python
# Just in case the Python environment doesn't contain pybind11, such as in
# a development build, we can fall back to FetchContent
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4 # Release 2.13.6
        FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(pybind11)


###############################################################################
#                                   pymodule                                  #
###############################################################################

python_add_library(RoughPy_PyModule MODULE WITH_SOABI)


set_target_properties(RoughPy_PyModule PROPERTIES
        LIBRARY_OUTPUT_NAME _roughpy
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(RoughPy_PyModule PRIVATE
        RPY_BUILDING_LIBRARY=1
        BOOST_UUID_FORCE_AUTO_LINK=1
)


set_target_properties(RoughPy_PyModule PROPERTIES ROUGHPY_COMPONENT PyModule)


if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    add_executable(test_python_embed)

    set_target_properties(test_python_embed PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )

    target_compile_definitions(test_python_embed PRIVATE
        RPY_BUILDING_LIBRARY=1
        BOOST_UUID_FORCE_AUTO_LINK=1
    )

    set_target_properties(test_python_embed PROPERTIES ROUGHPY_COMPONENT PyModule)
endif ()

#set_target_properties(RoughPy_PyModule PROPERTIES
#    INSTALL_RPATH $ORIGIN)
if (WIN32)

elseif (APPLE)
    #    set_target_properties(RoughPy_PyModule PROPERTIES
    #            INSTALL_RPATH "@loader_path;@loader_path/../../..")
else ()
    set_target_properties(RoughPy_PyModule PROPERTIES
            INSTALL_RPATH $ORIGIN)
endif ()

add_subdirectory(src)

target_include_directories(RoughPy_PyModule PRIVATE
        src/
)


target_link_libraries(RoughPy_PyModule PRIVATE
        pybind11::headers
        Boost::boost
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Intervals
        RoughPy::Algebra
        RoughPy::Streams
        RoughPy::PrecompiledHeaders

        #        $<LINK_LIBRARY:WHOLE_ARCHIVE,RoughPy::Streams>
        #        recombine::recombine
)


if (ROUGHPY_LINK_NUMPY)
    target_link_libraries(RoughPy_PyModule PRIVATE Python::NumPy)
    target_compile_definitions(RoughPy_PyModule PRIVATE
            ROUGHPY_WITH_NUMPY)
endif ()



if (WIN32)
    # Windows #pragma comment(lib, <libname>) links are such a pain.
    # Python includes such link statements, which makes it basically impossible
    # to link to it in a meaningful manner. To make things worse, Pybind11
    # disables the _DEBUG macro so it links to the release version
    # rather than the debug version when not using the minimal API.
    # This is a huge problem if one actually wants to link against
    # the debug version of the library. The fix is to just disable
    # all of them and let the build system do its job.

    foreach(_lib IN LISTS Python_LIBRARIES)
        if ("${_lib}" MATCHES "debug|optimized")
            continue()
        endif()
        cmake_path(GET _lib FILENAME _lib_name)
        message(DEBUG "Disable autolinking to ${_lib_name}")
        list(APPEND _disable_autolinking "/nodefaultlib:${_lib_name}")
    endforeach()

    # We don't use it, but just in case we do want to use the limited ABI
    list(APPEND _disable_autolinking "/nodefaultlib:python3.lib")

    target_link_options(RoughPy_PyModule PRIVATE ${_disable_autolinking})
endif()



configure_file(version.py.in ${CMAKE_CURRENT_LIST_DIR}/version.py @ONLY)


if (ROUGHPY_BUILD_TESTS)
    # If we're building tests, we need to make sure the _roughpy python module is located within
    # the roughpy directory so it can be imported correctly.
    if (WIN32)
        # Windows DLLs are dumb: they don't have a mechanism for locating their dependencies.
        # This is a major pain. To get around this, have to copy the runtime files from the build
        # tree into the directory alongside the _roughpy module. There might be a better way of doing
        # this by modifying the PATH variable as we did above, but it's just easier to copy everything
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_FILE:RoughPy_PyModule>
        )
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_RUNTIME_DLLS:RoughPy_PyModule>
                COMMAND_EXPAND_LISTS
        )
    else()
        # On sensible operating systems, we just need the shared _roughpy module to be placed directly
        # into the correct output directory.
        set_target_properties(RoughPy_PyModule PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    endif ()


    # TODO: This was a nice idea, but it doesn't work right. Come back to this later.
#    # If we already have pytest, we probably want to test the python alongside the
#    # C++ libraries
#    execute_process(COMMAND ${Python_EXECUTABLE} -m pytest --version
#        RESULT_VARIABLE _has_pytest
#            OUTPUT_QUIET ERROR_QUIET
#        )
#
#    if ("${_has_pytest}" EQUAL "0")
#        # There are python tests too
#        set(python_test_dir "${RoughPy_SOURCE_DIR}/tests")
#
#        add_test(
#                NAME PyTest
#                COMMAND ${Python_EXECUTABLE} -m pytest $<SHELL_PATH:${python_test_dir}>
#        )
#
#        set_property(TEST PyTest APPEND PROPERTY
#                ENVIRONMENT_MODIFICATION
#                    PYTHONPATH=path_list_append:"$<SHELL_PATH:${RoughPy_SOURCE_DIR}>"
#        )
#
#    endif()
#
#
#
endif()

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    target_include_directories(test_python_embed PRIVATE
        src/
    )

    target_link_libraries(test_python_embed PRIVATE
        pybind11::embed
        pybind11::headers
        Boost::boost
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Intervals
        RoughPy::Algebra
        RoughPy::Streams
        RoughPy::PrecompiledHeaders
    )

    target_sources(test_python_embed PRIVATE
        src/test_python_embed.cpp
    )

    if (ROUGHPY_LINK_NUMPY)
        target_link_libraries(test_python_embed PRIVATE Python::NumPy)
        target_compile_definitions(test_python_embed PRIVATE ROUGHPY_WITH_NUMPY)
    endif ()

   set_target_properties(test_python_embed
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_compile_definitions(test_python_embed
        PRIVATE
            BOOST_UUID_FORCE_AUTO_LINK=1
    )

    setup_roughpy_cpp_tests(test_python_embed)

    # _roughpy location fix as above if ROUGHPY_BUILD_TESTS block.
    if (WIN32)
        add_custom_command(TARGET test_python_embed POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_FILE:test_python_embed>
        )
    else ()
        set_target_properties(test_python_embed PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    endif ()
endif ()



add_subdirectory(compute)