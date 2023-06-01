
include(GNUInstallDirs)
include(GenerateExportHeader)
include(GoogleTest OPTIONAL)


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

function(add_roughpy_lib _name)
    cmake_parse_arguments(
            ARG
            "STATIC;SHARED;INTERFACE"
            ""
            "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS;DEFINITIONS;PUBLIC_HEADERS;PVT_INCLUDE_DIRS"
            ${ARGN}
    )

    _check_and_set_libtype(_lib_type ${ARG_SHARED} ${ARG_STATIC} ${ARG_INTERFACE})

    set(_real_name "RoughPy_${_name}")
    set(_alias_name "RoughPy::${_name}")
    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})

    if (NOT _lib_type STREQUAL INTERFACE)
        set(_private_include_dirs "${CMAKE_CURRENT_LIST_DIR}/include/roughpy/${_component}/")
        if (ARG_PVT_INCLUDE_DIRS)
            foreach(_pth IN LISTS ARG_PVT_INCLUDE_DIRS)
                list(APPEND _private_include_dirs ${CMAKE_CURRENT_LIST_DIR}/${_pth})
            endforeach ()
        endif ()
        foreach(_pth IN LISTS _private_include_dirs)
            if (NOT EXISTS "${_pth}")
                message(FATAL_ERROR "The path ${_pth} does not exist")
            endif ()
        endforeach ()
    endif ()

    add_library(${_real_name} ${_lib_type})
    add_library(${_alias_name} ALIAS ${_real_name})
    message(STATUS "Adding library ${_alias_name} version ${PROJECT_VERSION}")

    if (NOT ${_lib_type} STREQUAL INTERFACE)
        target_include_directories(${_real_name}
                PRIVATE
                "${_private_include_dirs}"
                PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
                )
        generate_export_header(${_real_name})
        target_sources(${_real_name}
                PUBLIC
                ${ARG_PUBLIC_HEADERS}
                PRIVATE
                ${ARG_SOURCES}
                )
        target_link_libraries(${_real_name}
                PUBLIC
                ${ARG_PUBLIC_DEPS}
                PRIVATE
                ${ARG_PRIVATE_DEPS}
                )
    else ()
        target_include_directories(${_real_name} INTERFACE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
                )
        target_sources(${_real_name} INTERFACE ${ARG_PUBLIC_HEADERS})
        target_link_libraries(${_real_name} INTERFACE ${ARG_PUBLIC_DEPS})
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

    if (_lib_type STREQUAL STATIC)
        set_target_properties(${_real_name} PROPERTIES
                POSITION_INDEPENDENT_CODE ON)
    endif ()
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
        else()
            target_sources(${_real_name} PRIVATE ${ARG_SOURCES})
        endif()
    endif()

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
    endif()

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
            "SRC;DEP;DEFN;COMPONENT_SRCS"
            ${ARGN}
    )

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component_name ${_component})
    set(_header_dir include/roughpy/${_component})

    set(_tests_name RoughPy_test_${_component}_${_name})
    message(DEBUG "Adding test ${_tests_name}")

    add_executable(${_tests_name} ${test_SRC} ${test_COMPONENT_SRCS})

    set(_deps)
    foreach (_dep IN LISTS test_DEP)
        if (TARGET ${_dep})
            list(APPEND _deps ${_dep})
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

    if (test_LINK_COMPONENT)
        string(CONCAT _component_name "RoughPy_" "${_component_name}")

        if (NOT TARGET ${_component_name})
            message(FATAL_ERROR "The target ${_component_name} does not exist")
        endif ()
        message(DEBUG "Linking component target ${_component_name}")
        target_link_libraries(${_tests_name} PRIVATE ${_component_name})
    endif ()


    target_compile_definitions(${_tests_name} PRIVATE ${tests_DEFN})
    target_include_directories(${_tests_name} PRIVATE
            ${_header_dir}
            ${CMAKE_CURRENT_BINARY_DIR})


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
    target_compile_definitions(${_lib_name} PRIVATE ${ARGS_DEFN})
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
        throw std::runtime_error(\"cannot get context\");
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


define_property(TARGET PROPERTY CONTEXT_INSTANCE_HEADER)
