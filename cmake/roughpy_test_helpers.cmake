


include(GoogleTest)


function(add_roughpy_test _name)
    if (ROUGHPY_BUILD_TESTS)
        cmake_parse_arguments(
                test
                "LINK_COMPONENT"
                ""
                "SRC;DEP;DEFN;COMPONENT_SRCS"
                ${ARGN}
        )

        cmake_path(GET CMAKE_CURRENT_SOURCE_DIR FILENAME _component)

        set(_header_dir include/roughpy/${_component})

        set(_tests_name RoughPy_test_${_component}_${_name})
        message(DEBUG "Adding test ${_tests_name}")

        add_executable(${_tests_name} ${test_SRC} ${test_COMPONENT_SRCS})

        target_link_libraries(${_tests_name} PRIVATE ${test_DEP} GTest::gtest_main)
        if (tests_LINK_COMPONENT)
            string(SUBSTRING ${_component} 0 1 _first_letter)
            string(SUBSTRING ${_component} 1 -1 _remaining)
            string(TOUPPER ${_first_letter} _first_letter)

            set(_component_name RoughPy_${_first_letter} ${_remaining})
            message(DEBUG "The component target is ${_component_name}")

            if (NOT TARGET ${_component_name})
                message(FATAL_ERROR "The target ${_component_name} does not exist")
            endif()
            target_link_libraries(${_tests_name} PRIVATE ${_component_name})
        endif()


        target_compile_definitions(${_tests_name} PRIVATE ${tests_DEFN})
        target_include_directories(${_tests_name} PRIVATE ${_header_dir})


        gtest_discover_tests(${_tests_name})
    endif ()
endfunction()
