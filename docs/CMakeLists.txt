
#find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

#foreach (_rpy_lib IN LISTS ROUGHPY_LIBS)
#    get_target_property(_lib_dir ${_rpy_lib} SOURCE_DIR)
#    list(APPEND DOXYGEN_INPUT_DIRS ${_lib_dir}/include)
#endforeach ()

#string(REPLACE ";" " " DOXYGEN_INPUT_DIR "${DOXYGEN_INPUT_DIRS}")
#set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
#set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
#
#set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
#set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
#

#add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
#         DEPENDS
#        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        MAIN_DEPENDENCY ${DOXYFILE_IN} ${DOXYFILE_OUT}
#        COMMENT "Generating docs with Doxygen")

#add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})


set(${CMAKE_CURRENT_BINARY_DIR}/applications)

include(FetchContent)

FetchContent_Declare(
        SigApplNotebooks
        GIT_REPOSITORY https://github.com/datasig-ac-uk/signature_applications
        GIT_TAG c4cc8303eb9eae88a5a25c50696d27b502d2217f
)

FetchContent_MakeAvailable(SigApplNotebooks)

FetchContent_GetProperties(SigApplNotebooks SOURCE_DIR SIG_APPS_NB_DIR)


file(GLOB_RECURSE NB_FILES "${SIG_APPS_NB_DIR}/**/*.ipynb")

set(NB_TOC_SECTIONS_LINES)
set(TOC_INDENT "   ")
foreach (_nb IN LISTS NB_FILES)
    cmake_path(GET _nb STEM _nb_name)
    list(APPEND NB_TOC_SECTIONS_LINES "${TOC_INDENT}${_nb_name}")
endforeach ()
list(JOIN NB_TOC_SECTIONS_LINES "\n" NB_TOC_SECTIONS)


configure_file(${CMAKE_CURRENT_LIST_DIR}/source/user/tutorials.rst.in ${CMAKE_CURRENT_LIST_DIR}/source/user/tutorials.rst @ONLY)


set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

add_custom_target(NoteBookFiles
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/nbexporter.py -o ${CMAKE_CURRENT_LIST_DIR}/source/user ${NB_FILES}
        DEPENDS ${NB_FILES}
        COMMAND_EXPAND_LISTS
)
add_custom_target(Sphinx ALL

        COMMAND ${CMAKE_COMMAND} -E rm -r ${SPHINX_BUILD}
        COMMAND ${SPHINX_EXECUTABLE} -b html
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        DEPENDS NoteBookFiles RoughPy_PyModule
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating docs with sphinx")

#add_custom_command(TARGET Sphinx
#        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E rm -r ${SPHINX_BUILD})

