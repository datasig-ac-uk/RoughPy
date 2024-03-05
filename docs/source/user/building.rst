.. _building-from-source:

Building from source
====================

RoughPy can be installed from source, although this is not the recommended way to install.
The build system requires `vcpkg <https://github.com/Microsoft/vcpkg>`_ in order to obtain the necessary dependencies (except for MKL on x86 platforms, which is installed via pip).
You will need to make sure that vcpkg is available on your system before attempting to build RoughPy.
The following commands should be sufficient to set up the environment for building RoughPy:

::

    git clone https://github.com/Microsoft/vcpkg.git tools/vcpkg
    tools/vcpkg/bootstrap-vcpkg.sh
    export CMAKE_TOOLCHAIN_FILE=$(pwd)/tools/vcpkg/scripts/buildsystems/vcpkg.cmake

With this environment variable set, most of the dependencies will be installed automatically during the build process.

On MacOS with Apple Silicon you will need to install libomp (for example using Homebrew ``brew install libomp``).
This is not necessary on Intel based MacOS where the Intel iomp5 can be used instead.
The build system will use ``brew --prefix libomp`` to try to locate this library.
(The actual ``brew`` executable can be customised by setting the ``ROUGHPY_BREW_EXECUTABLE`` CMake variable
or environment variable.)

You should now be able to pip install either using the PyPI source distribution (using the ``--no-binary :roughpy:``
flag), or directly from GitHub (recommended):

::

    pip install git+https://github.com/datasig-ac-uk/RoughPy.git

It will take some time to build.