# Build and debug in VSCode devcontainer

The `.devcontainer/` folder provides a consistent build and debug environment that can be run in VSCode locally or in GitHub Codespaces remotely.

## Opening devcontainer locally

When you open the cloned RoughPy folder in VSCode, accept the popup to open in a Dev Container or run the editor command `Dev Containers: Reopen in Dev Container`.

VSCode will restart inside a container that will have mapped your checkout folder to `/workspaces/<dirname>`, so be mindful that editing files is affecting this folder on your machine.

The devcontainer `Dockerfile` sets up a root `/tools` folder that contains a `venv` with build dependencies pre-installed, and a clone of `vcpkg`; both of which are added to `$PATH`.

## Building, testing and debugging

Run the editor command `CMake: Select Configure Preset` and select `Debug Develop` to compile with debug symbols. Then run `CMake: build`.

The devcontainer automatically installs the pytest and CTest extension, so after building you can launch any of the tests from the VSCode test explorer panel. Alternatively, you can run directly `pytest` directly from the integrated terminal.

VSCode loads debug scenarios from `.vscode/launch.json`. However, developers frequently need to edit this file with changes that should not be committed, so instead it is stored in `.devcontainer/example_launch.json`. Manually copy this over with:

```sh
    cp .devcontainer/example_launch.json .vscode/launch.json
```

Select a debug scenario in the Run and Debug panel, add a breakpoint on the code to stop on, and launch with F5.

The three example scenarios are:

- `debugpy pytest` - Debug a specific `pytest` unit test in a python debugger.
- `gdb pytest` - Start a python test and debug any underlying C++.
- `gdb gtest` - Debug a specific GTest unit test in a C++ debugger.
