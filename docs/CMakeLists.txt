
find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

foreach (_rpy_lib IN LISTS ROUGHPY_LIBS)
    get_target_property(_lib_dir ${_rpy_lib} SOURCE_DIR)
    list(APPEND DOXYGEN_INPUT_DIRS ${_lib_dir}/include)
endforeach ()

string(REPLACE ";" " " DOXYGEN_INPUT_DIR "${DOXYGEN_INPUT_DIRS}")
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)


add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        # DEPENDS
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        MAIN_DEPENDENCY ${DOXYFILE_IN} ${DOXYFILE_OUT}
        COMMENT "Generating docs with Doxygen")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})


set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)


add_custom_target(Sphinx ALL
        COMMAND ${SPHINX_EXECUTABLE} -b html
        -Dbreathe_projects.RoughPy=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        DEPENDS Doxygen
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating docs with sphinx")

