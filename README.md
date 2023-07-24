<center>
  <img src="https://raw.githubusercontent.com/datasig-ac-uk/RoughPy/main/branding/logo/logo_square_white.jpg">
</center><br>

# RoughPy
RoughPy is a package for working with streaming data as rough paths, and working with algebraic objects such as free tensors, shuffle tensors, and elements of the free Lie algebra.

This library is currently in an alpha stage, and as such many features are still incomplete or not fully implemented. Please bear this in mind when looking at the source code.


## Installation
RoughPy can be installed from PyPI using `pip` on Windows, Linux, and MacOS (Intel based Mac only, sorry not Apple Silicon support yet). Simply run
```
pip install roughpy
```
to get the latest version.

Alternatively, the wheel files can be downloaded from the [Releases](https://github.com/datasig-ac-uk/RoughPy/releases) page.

### Installing from source
RoughPy can be installed from source, although this is not the recommended way to install.
The build system requires [vcpkg](https://github.com/Microsoft/vcpkg) in order to obtain the necessary dependencies (except for MKL on x86 platforms, which is installed via pip).
You will need to make sure that vcpkg is available on your system before attempting to build RoughPy.
The following commands should be sufficient to set up the environment for building RoughPy:
```bash
git clone https://github.com/Microsoft/vcpkg.git tools/vcpkg
tools/vcpkg/bootstrap-vcpkg.sh
export CMAKE_TOOLCHAIN_FILE=$(pwd)/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
```
With this environment variable set, most of the dependencies will be installed automatically during the build process.

On MacOS with Apple Silicon you will need to install libomp (for example using Homebrew `brew install libomp`).
This is not necessary on Intel based MacOS where the Intel iomp5 can be used instead. 
The build system will use `brew --prefix libomp` to try to locate this library.
(The actual `brew` executable can be customised by setting the `ROUGHPY_BREW_EXECUTABLE` CMake variable 
or environment variable.)

You should now be able to pip install either using the PyPI source distribution (using the `--no-binary :roughpy:` 
flag), or directly from GitHub (recommended):
```bash
pip install git+https://github.com/datasig-ac-uk/RoughPy.git
```
It will take some time to build.

## Usage
Following the NumPy (and related) convention, we import RoughPy under the alias `rp` as follows:
```python
import roughpy as rp
```
The main object(s) that you will interact with are `Stream` objects or the family of factory classes such as `LieIncrementStream`. For example, we can create a `LieIncrementStream` using the following commands:
```python
stream = rp.LieIncrementStream.from_increments([[0., 1., 2.], [3., 4., 5.]], depth=2)
```
This will create a stream whose (hidden) underlying data are the two increments `[0., 1., 2.]` and `[3., 4., 5.]`, and whose algebra elements are truncated at maximum depth 2.
To compute the log signature over an interval we use the `log_signature` method on the stream, for example
```python
interval = rp.RealInterval(0., 1.)
lsig = stream.log_signature(interval)
```
Printing this new object `lsig` should give the following result
```
{ 1(2) 2(3) }
```
which is the first increment from the underlying data. (By default, the increments are assumed to occur at parameter values equal to their row index in the provided data.)

Similarly, the signature can be computed using the `signature` method on the stream object:
```python
sig = stream.signature(interval)
```
Notice that the `lsig` and `sig` objects have types `Lie` and `FreeTensor`, respectively. They behave exactly as you would expect elements of these algebras to behave. Moreover, they will (usually) be convertible directly to a NumPy array (or TensorFlow, PyTorch, JAX tensor type in the future) of the underlying data, so you can interact with them as if they were simple arrays.



## Support
If you have a specific problem, the best way to record this is to open an issue on GitHub. We welcome any feedback or bug reports.

## Contributing 
In the future, we will welcome pull requests to implement new features, fix bugs, add documentation or examples, or add tests to the project.
However, at present, we do not have robust CI pipelines set up to rigorously test incoming changes, and therefor will not be accepting pull requests made from outside the current team.


## Contributors
The full list of contributors is listed in `THANKS` alongside this readme. The people mentioned in this document constitute `The RoughPy Developers`.

## License
RoughPy is licensed under a BSD-3-Clause license. This was chosen specifically to match the license of NumPy.
