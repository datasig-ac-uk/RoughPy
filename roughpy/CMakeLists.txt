# This file is the main file for the Python module only. It should find all of
# the dependencies required for this module regardless of whether these might
# have been found by the surrounding build process.
# In particular, it needs to find Python, pybind11, and any other libraries
# that are specific to the Python library only. Also any setup that depends on
# SKBUILD can be done in here rather than relying on the top-level CMakeLists.

cmake_minimum_required(VERSION 3.21)


project(RoughPy_PyModule
        VERSION ${RoughPy_VERSION}
        LANGUAGES CXX
)
message(STATUS "Adding RoughPy::PyModule version ${RoughPy_VERSION}")




set(PYBIND11_FINDPYTHON ON)

if (NOT DEFINED pybind11_ROOT)
    execute_process(COMMAND
        "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
        RESULT_VARIABLE _python_pybind11_dir_found
        OUTPUT_VARIABLE _python_pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (_python_pybind11_dir_found EQUAL 0 AND EXISTS "${_python_pybind11_dir}")
        message(STATUS "Adding Pybind11 cmake dir ${_python_pybind11_dir}")
        set(pybind11_ROOT "${_python_pybind11_dir}" CACHE INTERNAL "")
    endif ()
endif ()



find_package(pybind11 REQUIRED)

if (NOT TARGET pybind11::headers)
    find_package(Pybind11 CONFIG REQUIRED)
endif ()


Python_add_library(RoughPy_PyModule MODULE WITH_SOABI)
set_target_properties(RoughPy_PyModule PROPERTIES
        LIBRARY_OUTPUT_NAME _roughpy
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
)

target_compile_definitions(RoughPy_PyModule PRIVATE
        RPY_BUILDING_LIBRARY=1
        BOOST_UUID_FORCE_AUTO_LINK=1
)

if (ROUGHPY_BUILD_PYMODULE_INPLACE)
    if (MSVC)
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_FILE:RoughPy_PyModule>
        )
        add_custom_command(TARGET RoughPy_PyModule POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy -t ${CMAKE_CURRENT_LIST_DIR} $<TARGET_RUNTIME_DLLS:RoughPy_PyModule>
            COMMAND_EXPAND_LISTS
        )
    else()
        set_target_properties(RoughPy_PyModule PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    endif ()
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

target_sources(RoughPy_PyModule PRIVATE
        src/roughpy_module.cpp
        src/roughpy_module.h
        #        src/recombine.cpp
        #        src/recombine.h
        src/algebra/algebra.h
        src/algebra/algebra.cpp
        src/algebra/algebra_iterator.cpp
        src/algebra/algebra_iterator.h
        src/algebra/basis.cpp
        src/algebra/basis.h
        src/algebra/context.cpp
        src/algebra/context.h
        src/algebra/free_multiply_funcs.cpp
        src/algebra/free_multiply_funcs.h
        src/algebra/free_tensor.h
        src/algebra/free_tensor.cpp
        src/algebra/lie.cpp
        src/algebra/lie.h
        src/algebra/lie_key.cpp
        src/algebra/lie_key.h
        src/algebra/lie_key_iterator.cpp
        src/algebra/lie_key_iterator.h
        src/algebra/lie_letter.cpp
        src/algebra/lie_letter.h
        src/algebra/setup_algebra_type.h
        src/algebra/shuffle_tensor.h
        src/algebra/shuffle_tensor.cpp
        src/algebra/tensor_key.cpp
        src/algebra/tensor_key.h
        src/algebra/tensor_key_iterator.cpp
        src/algebra/tensor_key_iterator.h
        src/args/convert_timestamp.cpp
        src/args/convert_timestamp.h
        src/args/dlpack_helpers.cpp
        src/args/dlpack_helpers.h
        src/args/kwargs_to_path_metadata.h
        src/args/kwargs_to_path_metadata.cpp
        src/args/kwargs_to_vector_construction.cpp
        src/args/kwargs_to_vector_construction.h
        src/args/numpy.cpp
        src/args/numpy.h
        src/args/parse_schema.cpp
        src/args/parse_schema.h
        src/args/strided_copy.cpp
        src/args/strided_copy.h
        src/intervals/date_time_interval.cpp
        src/intervals/date_time_interval.h
        src/intervals/dyadic.h
        src/intervals/dyadic.cpp
        src/intervals/dyadic_interval.cpp
        src/intervals/dyadic_interval.h
        src/intervals/interval.h
        src/intervals/interval.cpp
        src/intervals/intervals.cpp
        src/intervals/intervals.h
        src/intervals/partition.cpp
        src/intervals/partition.h
        src/intervals/real_interval.cpp
        src/intervals/real_interval.h
        src/intervals/segmentation.cpp
        src/intervals/segmentation.h
        src/scalars/parse_key_scalar_stream.cpp
        src/scalars/parse_key_scalar_stream.h
        src/scalars/r_py_polynomial.cpp
        src/scalars/r_py_polynomial.h
        #        src/scalars/scalar.h
        #        src/scalars/scalar.cpp
        src/scalars/scalar_type.h
        src/scalars/scalar_type.cpp
        src/scalars/scalars.cpp
        src/scalars/scalars.h
        src/streams/BaseStream.cpp
        src/streams/BaseStream.h
        src/streams/brownian_stream.cpp
        src/streams/brownian_stream.h
        src/streams/externally_sourced_stream.cpp
        src/streams/externally_sourced_stream.h
        src/streams/function_stream.cpp
        src/streams/function_stream.h
        src/streams/lie_increment_stream.h
        src/streams/lie_increment_stream.cpp
        src/streams/piecewise_abelian_stream.cpp
        src/streams/piecewise_abelian_stream.h
        src/streams/r_py_tick_construction_helper.cpp
        src/streams/r_py_tick_construction_helper.h
        src/streams/py_parametrization.cpp
        src/streams/py_parametrization.h
        src/streams/schema.cpp
        src/streams/schema.h
        src/streams/schema_finalization.cpp
        src/streams/schema_finalization.h
        src/streams/stream.cpp
        src/streams/stream.h
        src/streams/streams.cpp
        src/streams/streams.h
        src/streams/tick_stream.h
        src/streams/tick_stream.cpp
)

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
        #        $<LINK_LIBRARY:WHOLE_ARCHIVE,RoughPy::Streams>
        #        recombine::recombine
)


target_precompile_headers(RoughPy_PyModule PRIVATE
        <roughpy/platform/devices.h>
        <roughpy/platform/serialization.h>
        <roughpy/platform/filesystem.h>
        <roughpy/scalars/key_scalar_array.h>
        <roughpy/algebra/context.h>
        <pybind11/pybind11.h>
        <pybind11/functional.h>
        <pybind11/stl.h>
)

set(RPY_TARGETS RoughPy::Platform
        RoughPy::Scalars
        RoughPy::Algebra
        RoughPy::Intervals
        RoughPy::Streams)


if (ROUGHPY_LINK_NUMPY)
    target_link_libraries(RoughPy_PyModule PRIVATE Python::NumPy)
    target_compile_definitions(RoughPy_PyModule PRIVATE
            ROUGHPY_WITH_NUMPY)
    target_precompile_headers(RoughPy_PyModule PRIVATE <pybind11/numpy.h>)
endif ()


get_target_property(RoughPy_Core_VERSION RoughPy::Core VERSION)
get_target_property(RoughPy_Platform_VERSION RoughPy::Platform VERSION)
get_target_property(RoughPy_Scalars_VERSION RoughPy::Scalars VERSION)
get_target_property(RoughPy_Intervals_VERSION RoughPy::Intervals VERSION)
get_target_property(RoughPy_Algebra_VERSION RoughPy::Algebra VERSION)
get_target_property(RoughPy_Streams_VERSION RoughPy::Streams VERSION)


configure_file(${CMAKE_SOURCE_DIR}/cmake/version.py.in ${CMAKE_CURRENT_LIST_DIR}/version.py @ONLY)



# The old install logic was terrible. Let's do this properly

if (SKBUILD)
    # There really is no reason to install the Py module if you're not doing an
    # Skbuild build, so don't bother generating the code in this case

    if (DEFINED SKBUILD_PLATLIB_DIR)
        set(_install_loc "${SKBUILD_PLATLIB_DIR}/roughpy")
    else()
        set(_install_loc "roughpy")
    endif()

    set(RPY_INCLUDE_REGEXES)
    if (DEFINED VCPKG_ROOT)
        message("Install ${VCPKG_ROOT}")
        list(APPEND RPY_INCLUDE_REGEX "${VCPKG_ROOT}/.*")
    endif()
    if (DEFINED RoughPy_ROOT)
        list(APPEND RPY_INCLUDE_REGEX "${RoughPy_ROOT}/.*")
    endif()

    # We need to make sure the runtime artifacts from the component libs,
    # and all their dependencies are installed alongside the module itself.
    # Unusually for Windows, there is a very simple solution here: install the
    # TARGET_RUNTIME_DLLs. On Linux and Mac, we have to rely on other means
    install(TARGETS RoughPy_PyModule
        RUNTIME_DEPENDENCY_SET RPyModule_Depends
        LIBRARY DESTINATION ${_install_loc}
            NAMELINK_SKIP
        RUNTIME DESTINATION ${_install_loc}
        ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
        INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
        FRAMEWORK DESTINATION ${_install_loc}
        )

    install(FILES py.typed DESTINATION ${_install_loc})
    install(DIRECTORY . DESTINATION ${_install_loc}
    FILES_MATCHING PATTERN "*.py"
                   PATTERN "*.pyi"
                   PATTERN "src" EXCLUDE
                   PATTERN "*/__pycache__" EXCLUDE)


    message("Include ${RPY_INCLUDE_REGEXES}")
    install(RUNTIME_DEPENDENCY_SET RPyModule_Depends
        DESTINATION ${_install_loc}
        PRE_EXCLUDE_REGEXES
        # Linux system libraries
        [[libc\.\so\..*]]
        [[libgcc_s\.so\..*]]
        [[libm\.so\..*]]
        [[libstdc\+\+\.so\..*]]
        [[ld.*\.so.*]]
        # Windows system libraries (pre)
        [[api-ms-win-.*]]
        [[ext-ms-.*]]
        [[kernel32\.dll]]

        POST_EXCLUDE_REGEXES
        # On Windows, exclude anything in the in the system32 directory
        [[.*/system32/.*\.dll]]
        POST_INCLUDE_REGEXES
        ${RPY_INCLUDE_REGEXES}
    )





endif()