
include_guard()




set(ROUGHPY_LIBS CACHE INTERNAL "")
set(ROUGHPY_RUNTIME_DEPS CACHE INTERNAL "")


function(get_brew_prefix _out_var _package)

    cmake_parse_arguments(
            ""
            "VERBOSE"
            "BREW_EXECUTABLE"
            ""
            ${ARGN}
    )

    set(_executable brew)
    if (_BREW_EXECUTABLE)
        set(_executable "${_BREW_EXECUTABLE}")
    elseif (DEFINED ROUGHPY_HOMEBREW_EXECUTABLE)
        set(_executable "${ROUGHPY_HOMEBREW_EXECUTABLE}")
    elseif (DEFINED ENV{ROUGHPY_HOMEBREW_EXECUTABLE})
        set(_executable "$ENV{ROUGHPY_HOMEBREW_EXECUTABLE}")
    endif ()

    if (NOT _VERBOSE AND ENV{VERBOSE})
        set(_VERBOSE ON)
    endif ()

    set(_verbosity_flag "")
    if (_VERBOSE)
        set(_verbosity_flag "ECHO_OUTPUT_VARIABLE ECHO_ERROR_VARIABLE")
    endif ()
    message(DEBUG "Locating ${_package} brew installation prefix")

    execute_process(COMMANDS ${_executable}
            "--prefix" "${_package}"
            RESULT_VARIABLE _result
            OUTPUT_VARIABLE _out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ${_verbosity_flat}
    )

    if (_result EQUAL 0)
        message(STATUS "Brew located ${_package} at ${_out}")
        set(${_out_var} "${_out}" PARENT_SCOPE)
    else ()
        message(DEBUG "Could not locate ${_package} using Brew")
        set(${_out_var} "${_out_var}-NOTFOUND")
    endif ()
endfunction()


function(find_boost)

    cmake_parse_arguments("BOOST" "" "VERSION" "COMPONENTS" ${ARGN})

    foreach (lib IN LISTS BOOST_COMPONENTS)
        message(STATUS "finding boost library ${lib}")
        if (DEFINED BOOST_VERSION)
            find_package("boost_${lib}" "${BOOST_VERSION}" CONFIG REQUIRED
                    "${lib}"
            )
        else()
            find_package("boost_${lib}" CONFIG REQUIRED
                    "${lib}"
            )
        endif()

    endforeach ()

    add_library(Boost::boost ALIAS Boost::headers)

endfunction()




# Conditionally set the version of the library based on whether RoughPy component
# versioning is enabled.
function(set_library_version_properties _library)

    if (NOT ROUGHPY_NO_LIBRARY_VERSIONS)

        get_target_property(_comp ${_library} ROUGHPY_COMPONENT)
        if (NOT _comp)
            message(FATAL_ERROR "cannot set library version")
        endif()

        string(TOUPPER ${_comp} _upper)
        set(_ver_name ${ROUGHPY_${_upper}_VERSION})

        set_target_properties(${_library} PROPERTIES
            VERSION ${_ver_name}
        )
    endif()

endfunction()




function(_get_component_name _out_var _component)
    string(SUBSTRING ${_component} 0 1 _first_letter)
    string(SUBSTRING ${_component} 1 -1 _remaining)
    string(TOUPPER ${_first_letter} _first_letter)
    string(CONCAT _comp_name "${_first_letter}" "${_remaining}")
    set(${_out_var} ${_comp_name} PARENT_SCOPE)
endfunction()


function(_check_and_set_libtype _out _shared _static _interface)

    if (_shared)
        if (_static OR _interface)
            message(FATAL_ERROR "Library cannot be both SHARED and STATIC or INTERFACE")
        endif ()

        set(${_out} SHARED PARENT_SCOPE)

    elseif (_static)
        if (_interface)
            message(FATAL_ERROR "Library cannot be both STATIC and INTERFACE")
        endif ()

        set(${_out} STATIC PARENT_SCOPE)
    elseif (_interface)
        set(${_out} INTERFACE PARENT_SCOPE)
    else ()
        set(${_out} SHARED PARENT_SCOPE)
    endif ()
endfunction()

function(_check_runtime_component _library _out_var)
    get_target_property(_imported ${_library} IMPORTED)
    if (_imported)
        get_target_property(_imported_loc ${_library} IMPORTED_LOCATION)
        set(${_out_var} ${_imported_loc} PARENT_SCOPE)
    else ()
        set(${_out_var} "${_library}" PARENT_SCOPE)
    endif ()
endfunction()

function(_check_runtime_deps _out_var)
    set(_these_deps)
    foreach (_list IN LISTS ARGN)
        foreach (_dep IN LISTS _list)
            if (NOT _dep MATCHES RoughPy)
                get_target_property(_name ${_dep} NAME)
                get_target_property(_type ${_dep} TYPE)
                get_target_property(_imported ${_dep} IMPORTED)

                if (_imported AND (_type STREQUAL "SHARED_LIBRARY" OR _type STREQUAL "MODULE_LIBRARY"))
                    _check_runtime_component(${_name} _this_dep)
                    message(STATUS "Runtime dependency added: ${_name}")
                    list(APPEND _these_deps ${_this_dep})
                endif ()

                if (_name MATCHES "MKL" AND DEFINED MKL_THREAD_LIB)
                    message(STATUS "Runtime dependency added: ${MKL_THREAD_LIB}")
                    list(APPEND _these_deps "${MKL_THREAD_LIB}")
                endif ()
            endif ()
        endforeach ()
    endforeach ()
    list(REMOVE_DUPLICATES _these_deps)
    set(${_out_var} ${_these_deps} PARENT_SCOPE)
endfunction()


function(_split_rpy_deps _rpy_deps_var _nrpy_deps_var _deps_list)
    foreach (_dep IN LISTS ${_deps_list})
        if (_dep MATCHES "RoughPy")
            list(APPEND _rpy_deps ${_dep})
        else ()
            list(APPEND _nrpy_deps ${_dep})
        endif ()
    endforeach ()
    set(${_rpy_deps_var} ${_rpy_deps} PARENT_SCOPE)
    set(${_nrpy_deps_var} ${_nrpy_deps} PARENT_SCOPE)
endfunction()


function(target_link_components _target _visibility)
    if (ARGC EQUAL 0)
        message(FATAL_ERROR "Wrong number of arguments provided to target_link_components")
    endif ()
    if (NOT TARGET ${_target})
        message(FATAL_ERROR "Target ${_target} does not exist")
    endif ()


    foreach (_component IN LISTS ARGN)

        if (NOT _component MATCHES "^RoughPy")
            set(_component "RoughPy_${_component}")
        endif ()
        if (NOT TARGET ${_component})
            message(FATAL_ERROR "Component ${_component} is not a RoughPy component")
        endif ()

        get_target_property(_comp_type ${_component} TYPE)
        if (NOT _comp_type STREQUAL "STATIC_LIBRARY")
            target_link_libraries(${_target} ${_visibility} ${_component})
        else ()
            get_target_property(_include_dirs ${_component} INTERFACE_INCLUDE_DIRECTORIES)
            #            message("Including in ${_target}: ${_include_dirs}")
            target_include_directories(${_target} ${_visibility} ${_include_dirs})

            get_target_property(_link_libs ${_component} INTERFACE_LINK_LIBRARIES)
            #            message("Linking in ${_target}: ${_link_libs}")
            target_link_libraries(${_target} ${_visibility} ${_link_libs})

            target_link_libraries(${_target} PRIVATE ${_component})
            #                    $<LINK_LIBRARY:WHOLE_ARCHIVE,${_component}>)
        endif ()
    endforeach ()


endfunction()

function(_parse_defs_for_configure _args)

    list(LENGTH _args _arg_len)
    while (_arg_len GREATER 2)
        list(POP_FRONT _args _var_name)
        list(POP_FRONT _args _var_val)

        set(${_var_name} "${_var_val}" PARENT_SCOPE)
        math(EXPR _arg_len "${_arg_len} - 2")
    endwhile ()

    if (_arg_len EQUAL 1)
        message(FATAL_ERROR "arguments to DEFINE must be <NAME> <VALUE> pairs")
    endif ()

endfunction()


function(_do_configure_file _tgt _path_in _path_out _defines _atonly_flag
        _is_public)
    if (_defines)
        _parse_defs_for_configure("${_defines}")
    endif ()

    if (_atonly_flag)
        set(_atonly "@ONLY")
    else ()
        set(_atonly "")
    endif ()

    configure_file(${_path_in} ${_path_out} ${_atonly})

    target_sources(${_tgt} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${_path_out})

endfunction()


function(_configure_file _tgt _args)
    set(flag_params "ATONLY;IS_PUBLIC")
    set(single_arg_params "IN;OUT")
    set(multi_arg_params "DEFINE")

    list(POP_FRONT _args _sanity_check)
    if (NOT _sanity_check STREQUAL "FILE")
        message(FATAL_ERROR "Invalid arguments to configure")
    endif ()


    list(LENGTH _args _nargs)
    while (_nargs GREATER 0)
        list(FIND _args "FILE" _next_pos)
        list(SUBLIST _args 0 ${_next_pos} file_args)
        if (NOT _next_pos EQUAL -1)
            list(SUBLIST _args ${_next_pos} -1 _args)
            list(POP_FRONT _args)
        else ()
            set(_args "")
        endif ()

        cmake_parse_arguments(
                "FILE"
                "${flag_args}"
                "${single_arg_params}"
                "${multi_arg_params}"
                "${file_args}"
        )

        #        cmake_path(IS_ABSOLUTE FILE_IN _is_absolute)
        #        if (_is_absolute)
        #            set(_path_in "${FILE_IN}")
        #        else()
        #            set(_path_in "${CMAKE_CURRENT_LIST_DIR}/${FILE_IN}")
        #        endif()

        cmake_path(ABSOLUTE_PATH FILE_IN OUTPUT_VARIABLE _path_in)
        set(_path_out "${FILE_OUT}")

        if (NOT EXISTS ${_path_in})
            message(FATAL_ERROR "The source file ${_path_in} does not exist")
        endif ()

        message(STATUS "generating file \"${_path_out}\" from \"${_path_in}\"")
        _do_configure_file("${_tgt}"
                "${_path_in}"
                "${_path_out}"
                "${FILE_DEFINE}"
                "${FILE_ATONLY}"
                "${FILE_IS_PUBLIC}")

        list(LENGTH _args _nargs)

    endwhile ()


endfunction()

function(_parse_dependencies _private_var _interface_var)

    set(flag_params "")
    set(single_arg_params "")
    set(multi_arg_params "PUBLIC;PRIVATE;INTERFACE")
    cmake_parse_arguments("DEP"
            "${flag_params}"
            "${single_arg_params}"
            "${multi_arg_params}"
            ${ARGN}
    )

    # First do the public dependencies
    while (DEP_PUBLIC)
        unset(_next)
        list(POP_FRONT DEP_PUBLIC _dep)
        if (DEP_PUBLIC)
            list(GET DEP_PUBLIC 0 _next)
        endif ()

        if (_next STREQUAL "IF")
            # Dependency is conditional
            set(_negate OFF)

            list(POP_FRONT DEP_PUBLIC) # get rid of IF
            list(POP_FRONT DEP_PUBLIC _condition)
            if (_condition STREQUAL "NOT")
                list(POP_FRONT DEP_PUBLIC _condition)
                set(_negate ON)
            endif ()

            if ((_negate AND NOT ${_condition})
                    OR (NOT _negate AND ${_condition}))
                # public deps get added to both private and interface lists
                list(APPEND _interface_libs "${_dep}")
                list(APPEND _private_libs "${_dep}")
            endif ()
        else ()
            # unconditional dependency
            # public deps get added to both private and interface lists
            list(APPEND _interface_libs "${_dep}")
            list(APPEND _private_libs "${_dep}")
        endif ()
    endwhile ()

    # Next the private dependencies
    while (DEP_PRIVATE)
        unset(_next)
        list(POP_FRONT DEP_PRIVATE _dep)
        if (DEP_PRIVATE)
            list(GET DEP_PRIVATE 0 _next)
        endif ()
        if (_next STREQUAL "IF")
            # Dependency is conditional
            set(_negate OFF)

            list(POP_FRONT DEP_PRIVATE) # get rid of IF
            list(POP_FRONT DEP_PRIVATE _condition)
            if (_condition STREQUAL "NOT")
                list(POP_FRONT DEP_PRIVATE _condition)
                set(_negate ON)
            endif ()

            if ((_negate AND NOT "${_condition}")
                    OR (NOT _negate AND "${_condition}"))
                list(APPEND _private_libs "${_dep}")
            endif ()
        else ()
            # unconditional dependency
            list(APPEND _private_libs "${_dep}")
        endif ()
    endwhile ()

    # And finally the interface dependencies
    while (DEP_INTERFACE)
        unset(_next)
        list(POP_FRONT DEP_INTERFACE _dep)
        if (DEP_PUBLIC)
            list(GET DEP_PUBLIC 0 _next)
        endif ()
        if (_next STREQUAL "IF")
            # Dependency is conditional
            set(_negate OFF)

            list(POP_FRONT DEP_INTERFACE) # get rid of IF
            list(POP_FRONT DEP_INTERFACE _condition)
            if (_condition STREQUAL "NOT")
                list(POP_FRONT DEP_INTERFACE _condition)
                set(_negate ON)
            endif ()

            if ((_negate AND NOT "${_condition}") OR (
                    NOT _negate AND "${_condition}"))
                list(APPEND _private_libs "${_dep}")
            endif ()
        else ()
            # unconditional dependency
            list(APPEND _private_libs "${_dep}")
        endif ()
    endwhile ()

    set(${_private_var} "${_private_libs}" PARENT_SCOPE)
    set(${_interface_var} "${_interface_libs}" PARENT_SCOPE)
endfunction()


function(add_roughpy_component _name)

    set(flag_params "STATIC" "SHARED" "INTERFACE")
    set(single_arg_params "")
    set(multi_arg_params
            "SOURCES"
            "DEPENDENCIES"
            "PUBLIC_DEPS"
            "PRIVATE_DEPS"
            "DEFINITIONS"
            "CONFIGURE"
            "PUBLIC_HEADERS"
            "PVT_INCLUDE_DIRS"
            "NEEDS")

    cmake_parse_arguments(
            ARG
            "${flag_params}"
            "${single_arg_params}"
            "${multi_arg_params}"
            ${ARGN}
    )

    #    _check_and_set_libtype(_lib_type ${ARG_SHARED} ${ARG_STATIC} ${ARG_INTERFACE})
    if (ARG_INTERFACE)
        set(_lib_type INTERFACE)
    else ()
        set(_lib_type SHARED)
    endif ()

    _parse_dependencies(_private_deps _interface_deps "${ARG_DEPENDENCIES}")

    set(_real_name "RoughPy_${_name}")
    set(_alias_name "RoughPy::${_name}")

    string(TOUPPER "${_name}" _name_upper)
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})

    if (NOT _lib_type STREQUAL INTERFACE)
        set(_private_include_dirs
                "${CMAKE_CURRENT_LIST_DIR}/include/roughpy/${_component}/"
                "${CMAKE_CURRENT_LIST_DIR}/src")
        if (ARG_PVT_INCLUDE_DIRS)
            foreach (_pth IN LISTS ARG_PVT_INCLUDE_DIRS)
                list(APPEND _private_include_dirs ${CMAKE_CURRENT_LIST_DIR}/${_pth})
            endforeach ()
        endif ()

        foreach (_pth IN LISTS _private_include_dirs)
            if (NOT EXISTS "${_pth}")
                message(FATAL_ERROR "The path ${_pth} does not exist")
            endif ()
        endforeach ()
    else ()
        if (_private_deps)
            message(FATAL_ERROR
                    "INTERFACE library cannot have private dependencies")
        endif ()

    endif ()

    add_library(${_real_name} ${_lib_type})
    add_library(${_alias_name} ALIAS ${_real_name})
    message(STATUS "Adding ${_lib_type} library ${_alias_name} version ${PROJECT_VERSION}")

    if (ROUGHPY_LIBS)
        set(ROUGHPY_LIBS "${ROUGHPY_LIBS};${_real_name}" CACHE INTERNAL "" FORCE)
    else ()
        set(ROUGHPY_LIBS "${_real_name}" CACHE INTERNAL "" FORCE)
    endif ()


    _split_rpy_deps(_pub_rpy_deps _pub_nrpy_deps _interface_deps)
    _split_rpy_deps(_pvt_rpy_deps _pvt_nrpy_deps _private_deps)


    set_target_properties(${_real_name} PROPERTIES
            EXPORT_NAME ${_name})

    if (NOT ${_lib_type} STREQUAL "INTERFACE")
        target_compile_definitions(${_real_name} PRIVATE "RPY_COMPILING_${_name_upper}=1")
        target_include_directories(${_real_name}
                PRIVATE
                "${_private_include_dirs}"
        )
        target_sources(${_real_name}
                PUBLIC
                ${ARG_PUBLIC_HEADERS}
                PRIVATE
                ${ARG_SOURCES}
        )
        target_link_libraries(${_real_name}
                PRIVATE
                ${_pvt_nrpy_deps}
        )
        if (ROUGHPY_ENABLE_DBG_ASSERT)
            target_compile_definitions(${_real_name} PRIVATE RPY_DEBUG=1)
        endif ()

        foreach (_rpy_dep IN LISTS _pvt_rpy_deps)
            get_target_property(_dep_type ${_rpy_dep} TYPE)
            if (_dep_type STREQUAL "STATIC")
                target_link_libraries(${_real_name} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${_rpy_dep}>)
            else ()
                target_link_libraries(${_real_name} PRIVATE ${_rpy_dep})
            endif ()
        endforeach ()

        #target_compile_definitions(${_real_name} PRIVATE RPY_BUILDING_LIBRARY=1)


    else ()

        target_sources(${_real_name} INTERFACE ${ARG_PUBLIC_HEADERS})
    endif ()

    if (ARG_CONFIGURE)
        _configure_file(${_real_name} "${ARG_CONFIGURE}")
    endif ()

    if (_lib_type MATCHES "INTERFACE")
        set(_public INTERFACE)
    else ()
        set(_public PUBLIC)
    endif ()

    target_include_directories(${_real_name} ${_public}
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    target_link_libraries(${_real_name} ${_public} ${_pub_nrpy_deps})
    foreach (_rpy_dep IN LISTS _pub_rpy_deps)
        get_target_property(_dep_type ${_rpy_dep} TYPE)
        message(STATUS "Linking ${_dep_type} rpy ${_rpy_dep}")
        if (_dep_type STREQUAL "STATIC_LIBRARY")
            target_link_libraries(${_real_name} ${_public} $<LINK_LIBRARY:WHOLE_ARCHIVE,${_rpy_dep}>)
        else ()
            target_link_libraries(${_real_name} ${_public} ${_rpy_dep})
        endif ()
    endforeach ()

    unset(_runtime_deps)
    _check_runtime_deps(_runtime_deps ${_pub_nrpy_deps} ${_pvt_nrpy_deps})

    if (_runtime_deps)
        set_target_properties(${_real_name} PROPERTIES RUNTIME_DEPENDENCIES ${_runtime_deps})
    endif ()

    set_target_properties(${_real_name} PROPERTIES
            PUBLIC_HEADER "${ARGS_PUBLIC_HEADERS}"
            LINKER_LANGUAGE CXX
            CXX_DEFAULT_VISIBILITY hidden
            VISIBILITY_INLINES_HIDDEN ON
            VERSION "${PROJECT_VERSION}"
    )
    #    if (_lib_type STREQUAL SHARED)
    #        set_target_properties(${_real_name} PROPERTIES
    #                SOVERSION ${PROJECT_VERSION_MAJOR})
    #    endif ()

    #    if (_lib_type STREQUAL STATIC)
    #        set_target_properties(${_real_name} PROPERTIES INTERFACE_LINK_LIBRARIES_DIRECT)
    #    endif ()

    if (_lib_type STREQUAL STATIC OR _lib_type STREQUAL OBJECT)
        set_target_properties(${_real_name} PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
    elseif (_lib_type STREQUAL SHARED)
        generate_export_header(${_real_name})
        set_target_properties(${_real_name} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
    endif ()

    target_link_components(${_real_name} ${_public} ${ARG_NEEDS})

endfunction()


function(extend_roughpy_lib _name)
    cmake_parse_arguments(
            ARG
            ""
            ""
            "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS;DEFINITIONS;PUBLIC_HEADERS;PVT_INCLUDE_DIRS"
            ${ARGN}
    )

    set(_real_name "RoughPy_${_name}")
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})

    get_target_property(_lib_type ${_real_name} TYPE)

    if (ARG_SOURCES)
        if (_lib_type STREQUAL "INTERFACE_LIBRARY")
            target_sources(${_real_name} INTERFACE ${ARG_SOURCES})
        else ()
            target_sources(${_real_name} PRIVATE ${ARG_SOURCES})
        endif ()
    endif ()

endfunction()


function(add_roughpy_algebra _name)

    cmake_parse_arguments(
            "ARG"
            "DEVICE;BUNDLE;REQUIRED"
            "BASIS_NAME;BASIS_FILE;INTERFACE_FILE;IMPLEMENTATION_FILE"
            "BASIS_PROPERTIES"
    )

    # set up the names
    set(_basis_name "${_name}Basis")
    if (ARG_BASIS_NAME)
        set(_basis_name "${ARG_BASIS_NAME}")
    endif ()

    set(_interface_name "${_name}Interface")
    set(_impl_name "${_name}Implementation")


endfunction()

function(add_roughpy_test _name)
    if (NOT ROUGHPY_BUILD_TESTS)
        return()
    endif ()
    cmake_parse_arguments(
            test
            "LINK_COMPONENT"
            ""
            "SRC;DEP;DEFN;COMPONENT_SRCS;NEEDS"
            ${ARGN}
    )

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})
    set(_header_dir include/roughpy)

    set(_tests_name RoughPy_test_${_component}_${_name})
    message(STATUS "Adding test ${_tests_name}")

    add_executable(${_tests_name} ${test_SRC} ${test_COMPONENT_SRCS})

    set(_deps)
    foreach (_dep IN LISTS test_DEP)
        if (TARGET ${_dep})
            list(APPEND _deps ${_dep})
        elseif (_dep MATCHES "RoughPy::")
        elseif (MATCHES "^RoughPy_Testing::(.+)")
            set(_real_dep "RoughPy_${_component_name}_Testing_${CMAKE_MATCH_1}")
            if (NOT TARGET ${_real_dep})
                message(FATAL_ERROR "no target named ${_dep}")
            endif ()

            list(APPEND _deps ${_real_dep})
        else ()
            message(FATAL_ERROR "no target named ${_dep}")
        endif ()

    endforeach ()


    target_link_libraries(${_tests_name} PRIVATE ${_deps} GTest::gtest_main)

    target_compile_definitions(${_tests_name} PRIVATE ${test_DEFN})
    target_include_directories(${_tests_name} PRIVATE
            ${_header_dir}
            ${CMAKE_CURRENT_BINARY_DIR})

    target_link_components(${_tests_name} PRIVATE ${test_NEEDS})

    if (WIN32)
        add_custom_command(TARGET ${_tests_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${_tests_name}> $<TARGET_RUNTIME_DLLS:${_tests_name}>
                COMMAND_EXPAND_LISTS)
    endif ()

    gtest_discover_tests(${_tests_name})
endfunction()


function(add_roughpy_test_helper NAME)
    if (NOT ROUGHPY_BUILD_TESTS)
        return()
    endif ()

    cmake_parse_arguments(
            "ARGS"
            "STATIC;SHARED"
            ""
            "SRCS;DEPS;DEFN;OPTS;INCLUDES"
            ${ARGN}
    )
    if (ARGS_STATIC AND ARGS_SHARED)
        message(FATAL_ERROR "Invalid library type, must be either STATIC or SHARED")
    endif ()

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component ${_component})
    set(_lib_name "RoughPy_${_component}_Testing_${_name}")
    if (DEFINED ARGS_STATIC)
        add_library(${_lib_name} STATIC)
    else ()
        add_library(${_lib_name} SHARED)
    endif ()

    set_target_properties(${_lib_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)

    target_sources(${_lib_name} PRIVATE ${ARGS_SRCS})

    target_link_libraries(${_lib_name} PRIVATE
            GTest::gtest
            ${ARGS_DEPS})

    target_include_directories(${_lib_name} PRIVATE ${ARGS_INCLUDES})
    target_compile_definitions(${_lib_name}
            PRIVATE ${ARGS_DEFN})

    if (NOT WIN32)
        target_compile_definitions(${_lib_name} PRIVATE RPY_BUILDING_LIBRARY=1)
    endif ()
    target_compile_options(${_lib_name} PRIVATE ${ARGS_OPTS})
endfunction()


