

include_guard()

define_property(TARGET PROPERTY ROUGHPY_COMPONENT)




macro(_do_set_variable _var_name _value _cache_descr)
    set(ROUGHPY_${_upper_name}_${_var_name} ${_value} CACHE FILEPATH "${_cache_descr}")
    message(DEBUG "    ROUGHPY_${_upper_name}_${_var_name}=${ROUGHPY_${_upper_name}_${_var_name}}")
endmacro()


function(setup_roughpy_component _name)

    cmake_parse_arguments(comp "" "" "VERSION" "" ${ARGN})

    string(TOUPPER ${_name} _upper_name)
    string(TOLOWER ${_name} _lower_name)

    if (NOT comp_VERSION)
        set(comp_VERSION ${RoughPy_VERSION})
    endif()


    message(STATUS "Adding RoughPy component ${_name} version ${comp_VERSION}")

    message(DEBUG "Setting variables for component ${_name}")
    _do_set_variable(VERSION ${comp_VERSION} "The version of the component")
    _do_set_variable(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}"
            "Root directory of the ${_name} component")
    _do_set_variable(INCLUDE_DIR "${ROUGHPY_${_upper_name}_ROOT_DIR}/include/roughpy/"
            "Include directory for the ${_name} component")
    _do_set_variable(INCLUDE "${ROUGHPY_${_upper_name}_INCLUDE_DIR}/${_lower_name}"
            "Include directory for the ${_name} component")
    _do_set_variable(SOURCE "${ROUGHPY_${_upper_name}_ROOT_DIR}/src"
            "Main source directory for the ${_name} component")



endfunction()
