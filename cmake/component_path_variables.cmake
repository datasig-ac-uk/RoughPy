

include_guard()




function(set_component_path_variables _component)
    if (NOT _component)
        message(FATAL_ERROR "_missing component")
    endif()

    string(TOUPPER ${_component} _name)
    string(TOLOWER ${_component} _lower_name)

    set(ROUGHPY_${_name}_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR} PARENT_SCOPE)
    set(ROUGHPY_${_name}_INCLUDE_DIR ${ROUGHPY_${_name}_ROOT_DIR}/include/roughpy PARENT_SCOPE)
    set(ROUGHPY_${_name}_${_name}_INCLUDE ${ROUGHPY_${_name}_INCLUDE_DIR}/${_lower_name} PARENT_SCOPE)
endfunction()
