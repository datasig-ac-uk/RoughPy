


include(GoogleTest)

function(_get_component_name _out_var _dir_name)
    string(SUBSTRING ${_component} 0 1 _first_letter)
    string(SUBSTRING ${_component} 1 -1 _remaining)
    string(TOUPPER ${_first_letter} _first_letter)
    string(CONCAT _comp_name "${_first_letter}" "${_remaining}")
    set(${_out_var} ${_comp_name} PARENT_SCOPE)
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
    foreach(_dep IN ${test_DEP})
        if (TARGET ${_dep})
            list(APPEND _deps ${_dep})
        elseif(MATCHES "^RoughPy_Testing::(.+)")
            set(_real_dep "RoughPy_${_component_name}_Testing_${CMAKE_MATCH_1}")
            if (NOT TARGET ${_real_dep})
                message(FATAL_ERROR "no target named ${_dep}")
            endif()

            list(APPEND _deps ${_real_dep})
        else()
            message(FATAL_ERROR "no target named ${_dep}")
        endif()

    endforeach()

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
    endif()

    cmake_parse_arguments(
            "ARGS"
            "STATIC;SHARED"
            ""
            "SRCS;DEPS;DEFN;OPTS;INCLUDES"
            ${ARGN}
    )
    if (ARGS_STATIC AND ARGS_SHARED)
        message(FATAL_ERROR "Invalid library type, must be either STATIC or SHARED")
    endif()

    cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)
    _get_component_name(_component ${_component})
    set(_lib_name "RoughPy_${_component}_Testing_${_name}")
    if (DEFINED ARGS_STATIC)
        add_library(${_lib_name} STATIC)
    else()
        add_library(${_lib_name} SHARED)
    endif()

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
