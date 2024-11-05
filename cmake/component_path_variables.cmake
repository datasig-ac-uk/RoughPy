

include_guard()


macro(_do_set_variable _var_name _value _cache_descr)
    set(ROUGHPY_${_name}_${_var_name} ${_value} CACHE FILEPATH "${_cache_descr}")
    message(DEBUG "    ROUGHPY_${_name}_${_var_name}=${ROUGHPY_${_name}_${_var_name}}")
endmacro()

function(set_component_path_variables _component)
    if (NOT _component)
        message(FATAL_ERROR "_missing component")
    endif()

    string(TOUPPER ${_component} _name)
    string(TOLOWER ${_component} _lower_name)

    message(DEBUG "Setting variables for component ${_component}")
    _do_set_variable(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}"
            "Root directory of the ${_component} component")
    _do_set_variable(INCLUDE_DIR "${ROUGHPY_${_name}_ROOT_DIR}/include/roughpy/${_lower_name}"
            "Include directory for the ${_component} component")
    _do_set_variable(INCLUDE "${ROUGHPY_${_name}_INCLUDE_DIR}"
            "Include directory for the ${_component} component")


endfunction()
