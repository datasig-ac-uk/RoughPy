# Find python and set Python_ variables in scope of calling CMakeLists.txt
# Any arguments passed will be loaded python components, e.g.
#    find_python_with_components(Interpreter)
#
function(find_python_with_components)
    find_package(Python 3.8 REQUIRED COMPONENTS ${ARGV})

    foreach(var IN ITEMS
        Python_EXECUTABLE
        Python_VERSION
        Python_VERSION_MAJOR
        Python_VERSION_MINOR
        Python_INCLUDE_DIRS
        Python_LIBRARIES
        Python_LIBRARY
        Python_Interpreter_FOUND
        Python_Development_FOUND
    )
        if(DEFINED ${var})
            set(${var} "${${var}}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()
