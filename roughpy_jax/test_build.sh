#!/usr/bin/env bash

# FIXME experimental test build of roughpy-jax wheel
#
# Python build cannot access files in the parent folder so cmake/ and roughpy_compute/
# directories are copied in here into _deps folder used in CMakeLists.txt.
#
# Possible workarounds:
# - Move roughpy_jax into a separate repo and bring in RoughPy as a submodule or CMake find
# - Remove roughpy_jax/pyproject.toml and instead build roughpy-jax from root project

echo "Copying python build deps..."
rm -rf _deps
mkdir _deps
cp ../cmake/find_python.cmake _deps
cp -r ../roughpy_compute _deps

echo "Building wheel..."
python -m build --wheel

echo "Done!"
