
include_guard()

include(GNUInstallDirs)
include(GenerateExportHeader)
include(GoogleTest OPTIONAL)


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
        set(${_out_var} "$<TARGET_FILE:${_library}>" PARENT_SCOPE)
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

function(add_roughpy_component _name)
    cmake_parse_arguments(
            ARG
            "STATIC;SHARED;INTERFACE"
            ""
            "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS;DEFINITIONS;PUBLIC_HEADERS;PVT_INCLUDE_DIRS;NEEDS"
            ${ARGN}
    )

    #    _check_and_set_libtype(_lib_type ${ARG_SHARED} ${ARG_STATIC} ${ARG_INTERFACE})
    if (ARG_INTERFACE)
        set(_lib_type INTERFACE)
    else ()
        set(_lib_type STATIC)
    endif ()


    set(_real_name "RoughPy_${_name}")
    set(_alias_name "RoughPy::${_name}")
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})

    if (NOT _lib_type STREQUAL INTERFACE)
        set(_private_include_dirs "${CMAKE_CURRENT_LIST_DIR}/include/roughpy/${_component}/")
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
    endif ()

    add_library(${_real_name} ${_lib_type})
    add_library(${_alias_name} ALIAS ${_real_name})
    message(STATUS "Adding ${_lib_type} library ${_alias_name} version ${PROJECT_VERSION}")

    if (ROUGHPY_LIBS)
        set(ROUGHPY_LIBS "${ROUGHPY_LIBS};${_real_name}" CACHE INTERNAL "" FORCE)
    else ()
        set(ROUGHPY_LIBS "${_real_name}" CACHE INTERNAL "" FORCE)
    endif ()

    _split_rpy_deps(_pub_rpy_deps _pub_nrpy_deps ARG_PUBLIC_DEPS)
    _split_rpy_deps(_pvt_rpy_deps _pvt_nrpy_deps ARG_PRIVATE_DEPS)

    set_target_properties(${_real_name} PROPERTIES
            EXPORT_NAME ${_name})

    if (NOT ${_lib_type} STREQUAL "INTERFACE")
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

        foreach (_rpy_dep IN LISTS _pvt_rpy_deps)
            get_target_property(_dep_type ${_rpy_dep} TYPE)
            if (_dep_type STREQUAL "STATIC")
                target_link_libraries(${_real_name} PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,${_rpy_dep}>)
            else ()
                target_link_libraries(${_real_name} PRIVATE ${_rpy_dep})
            endif ()
        endforeach ()

        target_compile_definitions(${_real_name} PRIVATE RPY_BUILDING_LIBRARY=1)


    else ()

        target_sources(${_real_name} INTERFACE ${ARG_PUBLIC_HEADERS})
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
    _check_runtime_deps(_runtime_deps ${ARG_PUBLIC_DEPS} ${ARG_PRIVATE_DEPS})

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


function(_compute_la_ctx_depths _width _check_size _out_var)
    if (_width EQUAL 1)
        set(${_out_var} "2;3;4;5;6" PARENT_SCOPE)
        return()
    endif ()

    math(EXPR tmp "(${_width} - 1) * ${_check_size} - 1")
    set(max_depth 0)

    while (tmp GREATER 0)
        math(EXPR tmp "${tmp} / ${_width}")
        math(EXPR max_depth "${max_depth} + 1")
    endwhile ()

    set(_degs)
    foreach (_depth RANGE 1 ${max_depth})
        list(APPEND _degs ${_depth})
    endforeach ()

    set(${_out_var} ${_degs} PARENT_SCOPE)

endfunction()


function(_handle_la_context _target _width _depth _ctype _class_name _main_header)


    get_target_property(_bin_dir ${_target} BINARY_DIR)
    get_target_property(_tname ${_target} NAME)

    get_target_property(_header_name ${_target} CONTEXT_INSTANCE_HEADER)

    string(TOUPPER ${_tname} _export_name)
    set(_export_name "${_export_name}_EXPORT")
    string(TOLOWER ${_tname} _export_header_name)
    set(_export_header_name "${_export_header_name}_export.h")

    set(_inline_config_names "${_tname}_contexts.inl")
    if (NOT _header_name)
        set(_header_name "${_tname}_contexts.h")
        set_target_properties(${_target} PROPERTIES CONTEXT_INSTANCE_HEADER ${_header_name})

        file(WRITE "${_bin_dir}/${_header_name}" "\
#include \"${_main_header}\"
#include \"${_export_header_name}\"
#include <libalgebra/coefficients.h>
namespace rpy { namespace algebra {
class ${_export_name} ${_class_name}Maker : public ContextMaker {
public:
    using typename ContextMaker::preference_list;

    bool can_get(deg_t width, deg_t depth, const scalars::ScalarType* ctype,
                 const preference_list& preferences) const override;
    context_pointer get_context(deg_t width, deg_t depth, const scalars::ScalarType* ctype,
                                const preference_list& preferences) const override;
    optional<base_context_pointer> get_base_context(deg_t width, deg_t depth) const override;

};
}}
")

        target_include_directories(${_target} PRIVATE ${_bin_dir})

        file(WRITE "${_bin_dir}/${_inline_config_names}" "")

        find_package(Boost 1.60 COMPONENTS headers REQUIRED)
        target_link_libraries(${_target} PRIVATE Boost::boost)

        file(WRITE "${_bin_dir}/${_class_name}Maker.cpp" "\
#include \"${_main_header}\"
#include \"${_header_name}\"
#include <boost/container/flat_map.hpp>

#include <utility>

using namespace rpy;
static const boost::container::flat_map<std::tuple<deg_t, deg_t, const scalars::ScalarType*>, algebra::context_pointer> s_la_contexts {
#include \"${_inline_config_names}\"
};

bool algebra::${_class_name}Maker::can_get(deg_t width, deg_t depth, const scalars::ScalarType* ctype, const preference_list& preferences) const {
    return s_la_contexts.find(std::make_tuple(width, depth, ctype)) != s_la_contexts.end();
}

algebra::context_pointer algebra::${_class_name}Maker::get_context(deg_t width, deg_t depth, const scalars::ScalarType* ctype, const preference_list& preferences) const {
    auto found = s_la_contexts.find(std::make_tuple(width, depth, ctype));
    if (found == s_la_contexts.end()) {
        RPY_THROW(std::runtime_error, \"cannot get context\");
    }
    return found->second;
}

optional<algebra::base_context_pointer> algebra::${_class_name}Maker::get_base_context(deg_t width, deg_t depth) const {
    for (const auto& item : s_la_contexts) {
        if (std::get<0>(item.first) == width && std::get<1>(item.first) == depth) {
            return static_cast<algebra::base_context_pointer>(item.second);
        }
    }
    return {};
}

"
                )

        target_sources(${_target} PRIVATE
                ${_header_name}
                ${_bin_dir}/${_class_name}Maker.cpp
                )
    endif ()


    set(_file_name "${_bin_dir}/${_tname}_${_width}_${_depth}_${_ctype}.cpp")

    file(WRITE ${_file_name} "\
#include \"${_header_name}\"
#include <libalgebra/libalgebra.h>

namespace rpy { namespace algebra {
template class ${_class_name}<${_width}, ${_depth}, ${RPY_LA_CTYPE_${_ctype}}>;
}}
")

    target_sources(${_target} PRIVATE ${_file_name})

    file(APPEND "${_bin_dir}/${_header_name}" "\
namespace rpy { namespace algebra {
extern template class ${_export_name} ${_class_name}<${_width}, ${_depth}, ${RPY_LA_CTYPE_${_ctype}}>;
}}

")

    file(APPEND "${_bin_dir}/${_inline_config_names}"
            "{{${_width}, ${_depth}, scalars::ScalarType::of<${RPY_CTYPE_${_ctype}}>()}, new rpy::algebra::${_class_name}<${_width}, ${_depth}, ${RPY_LA_CTYPE_${_ctype}}>()},
"
            )


endfunction()

function(add_libalgebra_contexts _name)

    set(_target_name "RoughPy_${_name}")
    if (NOT TARGET ${_target_name})
        message(FATAL_ERROR "Target ${_name} not found")
    endif ()


    cmake_parse_arguments(
            "ARG"
            ""
            "CHECK_SIZE;CLASS_NAME;CLASS_HEADER"
            "WIDTH;DEPTH;COEFFS"
            ${ARGN}
    )


    if (NOT ARG_WIDTH)
        message(FATAL_ERROR "No widths specified")
    endif ()


    if (ARG_CLASS_NAME)
        set(_class_name ${ARG_CLASS_NAME})
    else ()
        set(_class_name LAContext)
    endif ()

    if (ARG_CLASS_HEADER)
        set(_class_header ${ARG_CLASS_HEADER})
    else ()
        set(_class_header "roughpy/la_context.h")
    endif ()

    unset(_ctypes)
    if (ARG_COEFFS)
        set(_ctypes ${ARG_COEFFS})
    else ()
        set(_ctypes "DPReal;SPReal")
    endif ()

    foreach (_ctype IN LISTS _ctypes)
        foreach (_width IN LISTS ARG_WIDTH)
            unset(_depths)
            if (ARG_DEPTH)
                set(_depths ${ARG_DEPTH})
            elseif (DEFINED "RPY_LACTX_W${_width}_DEPTHS")
                set(_depths ${RPY_LACTX_W${_width}_DEPThS})
            elseif (ARG_CHECK_SIZE)
                math(EXPR _size "${ARG_CHECK_SIZE} / ${RPY_SIZEOF_${_ctype}}")
                _compute_la_ctx_depths(${_width} ${_size} _depths)
            else ()
                continue()
            endif ()

            message(STATUS "Adding Width ${_width}, stype ${_ctype}, Depths: ${_depths}")

            foreach (_depth IN LISTS _depths)
                _handle_la_context(${_target_name} ${_width} ${_depth} ${_ctype} ${_class_name} ${_class_header})
            endforeach ()


        endforeach ()
    endforeach ()


endfunction()


set(RPY_SIZEOF_DPReal 8 CACHE STRING "Sizeof double precision real number")
set(RPY_SIZEOF_SPReal 4 CACHE STRING "Sizeof single precision real number")
set(RPY_LA_CTYPE_DPReal alg::coefficients::double_field CACHE STRING "Name of double precision coeff ring")
set(RPY_LA_CTYPE_SPReal alg::coefficients::float_field CACHE STRING "Name of single precision coeff ring")
set(RPY_CTYPE_DPReal double)
set(RPY_CTYPE_SPReal float)


define_property(TARGET PROPERTY CONTEXT_INSTANCE_HEADER BRIEF_DOCS "Context header" FULL_DOCS "Docs ")
