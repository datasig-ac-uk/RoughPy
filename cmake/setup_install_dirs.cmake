
include_guard()

# If we're building with SKBUILD, we need to define install locations for
# all the components using their special directory variables. Otherwise,
# use the GNUInstall dirs
if (SKBUILD)

    # This is all the variables set by GNUInstallDirs, minus LIBDIR and BINDIR
    set(_ignore_dirs
            SBINDIR
            LIBEXECDIR
            SYSCONFIGDIR
            SHAREDSTATEDIR
            LOCALSTATEDIR
            RUNSTATEDIR
            INCLUDEDIR
            OLDINCLUDEDIR
            DATAROOTDIR
            DATADIR
            INFODIR
            LOCALEDIR
            MANDIR
            DOCDIR
    )

    if (WIN32)
        # On Windows, DLLs are put in BINDIR
        list(APPEND _ignore_dirs LIBDIR)
        set(CMAKE_INSTALL_BINDIR ${SKBUILD_PLATLIB_DIR}/roughpy CACHE STRING
                "Overwritten install for BINDIR")
    else ()
        # On not Windows, Shared Objects go in LIBDIR
        list(APPEND _ignore_dirs BINDIR)
        set(CMAKE_INSTALL_LIBDIR ${SKBUILD_PLATLIB_DIR}/roughpy CACHE STRING
                "Overwritten install for LIBDIR")

        list(APPEND _ignore_dirs BINDIR)
    endif ()

    foreach (_dir ${_ignore_dirs})
        set(CMAKE_INSTALL_${_dir} ${SKBUILD_NULL_DIR} CACHE STRING
                "Overwritten install for ${_dir}")
    endforeach ()

else ()
    include(GNUInstallDirs)
endif ()
