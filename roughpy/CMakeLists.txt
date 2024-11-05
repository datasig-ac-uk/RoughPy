

setup_roughpy_component(PyModule)

###############################################################################
#                                    python                                   #
###############################################################################
# We need to provide some help to make sure we find the correct version of
# Python. Ideally, if we're using Scikit-Build-Core to build the library (via
# pip) and the Python executable is provided via the PYTHON_EXECUTABLE cache
# variable. In this case, make sure that this is the version of Python that gets
# found.
set(Python_FIND_VIRTUAL_ENVIRONMENT FIRST)
if (NOT PYTHON_FOUND AND SKBUILD)
    cmake_path(GET PYTHON_EXECUTABLE PARENT_PATH _sk_env_dir)
    message(STATUS "SKBuild environment: ${_sk_env_dir}")

    # Some variables that are set might cause issues in the FindPython module,
    # so unset those
    unset(Python_LIBRARY CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(Python3_LIBRARY CACHE)

    # clean up temporary
    unset(_sk_env_dir)
else ()
    # If we're not using Scikit-Build-Core (i.e. a pure CMake build) then try
    # looking for a Python virtual environment first.
    set(Python_FIND_VIRTUALENV FIRST)

    # In particular, if ENV{VIRTUAL_ENV} is set then add this to the cmake
    # prefix path so FindPython is more likely to find this environemnt.
    if (DEFINED ENV{VIRTUAL_ENV})
        # Put venv/lib on the prefix path so we can find
        # a pip installed MKL
        message(STATUS "Adding python virtual environment to path")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}")
    endif ()
    if (DEFINED ROUGHPY_PYTHON_VENV_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${ROUGHPY_PYTHON_VENV_DIR}")
    endif()
endif ()


set(_python_components Interpreter Development.Module)
if (ROUGHPY_LINK_NUMPY)
    list(APPEND _python_components NumPy)
endif ()


find_package(Python 3.8 REQUIRED COMPONENTS ${_python_components})
unset(_python_components)



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

set_target_properties(RoughPy_PyModule PROPERTIES ROUGHPY_COMPONENT PyModule)

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
        src/args/buffer_info.cpp
        src/args/buffer_info.h
        src/args/check_for_excess_args.cpp
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
        src/args/parse_algebra_configuration.cpp
        src/args/parse_algebra_configuration.h
        src/args/parse_data_argument.cpp
        src/args/parse_data_argument.h
        src/args/parse_schema.cpp
        src/args/parse_schema.h
        src/scalars/pytype_conversion.cpp
        src/scalars/pytype_conversion.h
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


configure_file(version.py.in ${CMAKE_CURRENT_LIST_DIR}/version.py @ONLY)



install(TARGETS RoughPy_PyModule
    RUNTIME DESTINATION roughpy
    LIBRARY
    DESTINATION roughpy
    NAMELINK_SKIP
    ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
    COMPONENT Development
    EXCLUDE_FROM_ALL
    INCLUDES DESTINATION ${SKBUILD_NULL_DIR}
    COMPONENT Development
        FRAMEWORK DESTINATION roughpy
    EXCLUDE_FROM_ALL
)

install(FILES roughpy/py.typed DESTINATION roughpy)
install(DIRECTORY roughpy
        DESTINATION .
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.pyi"
        PATTERN "src/*" EXCLUDE)





if (ROUGHPY_BUILD_TESTS)

    # There are python tests too





endif()
