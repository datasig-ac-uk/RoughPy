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

## Intervals in RoughPy
RoughPy is very careful in how it works with intervals.
One design goal is that it should be able to handle jumps in the underlying signal that occur at particular times, including the beginning or end of the interval, and still guarantee that if you combine the signature over adjacent interval, you always get the signature over the entire interval.
This implies that there has to be a decision about whether data at the exact beginning or exact end of the interval is included.
The convention in RoughPy are that we use clopen intervals, and that data at beginning of the interval is seen, and data at the end of the interval is seen in the next interval.
A second design goal is that the code should be efficient, and so the internal representation of a stream involves caching the signature over dyadic intervals of different resolutions.
Recovering the signature over any interval using the cache has logarithmic complexity (using at most 2n tensor multiplications, when n is the internal resolution of the stream).
Resolution refers to the length of the finest granularity at which we will store information about the underlying data.
Any event occurs within one of these finest granularity intervals, multiple events occur within the same interval resolve to a more complex log-signature which correctly reflects the time sequence of the events within this grain of time.
However, no query of the stream is allowed to see finer resolution than the internal resolution of the stream, it is only allowed to access the information over intervals that are a union of these finest resolution granular intervals.
For this reason, a query over any interval is replaced by a query is replaced by a query over an interval whose endpoints have been shifted to be consistent with the granular resolution, obtained by rounding these points to the contained end-point of the unique clopen granular interval containing this point.
In particular, if both the left-hand and right-hand ends of the interval are contained in the clopen granular interval, we round the interval to the empty interval. 
Specifying a resolution of 32 or 64 equates to using integer arithmetic.

## Usage
Following the NumPy (and related) convention, we import RoughPy under the alias `rp` as follows:
```python
import roughpy as rp
```
The main object(s) that you will interact with are `Stream` objects or the family of factory classes such as `LieIncrementStream`. For example, we can create a `LieIncrementStream` using the following commands:
```python
import numpy as np
stream = rp.LieIncrementStream.from_increments(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64), depth=2)
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

We can also construct streams by providing the raw data of Lie increments with higher order terms by specifying the width using the same constructor above.
For example, if we take width 2 and depth 2 then the elements of a Lie element will have keys (1, 2, [1,2]).
So if we provide the following data, we construct a stream whose underlying Lie increments are width 2, depth 2 
```
stream = rp.LieIncrementStream.from_increments(np.array([[1, 2, 0.5], [0.2, -0.1, 0.2]], dtype=np.float64), width=2, depth=2)
print(stream.log_signature(rp.RealInterval(0, 0.5), 2)) # returns the first increment
# { 1(1), 2(2), 0.5([1,2]) }
>>> print(stream.log_signature(rp.RealInterval(0, 1.5), 2)) # Campbell-Baker-Hausdorff product of both increments
{ 1.2(1) 1.9(2) 0.45([1,2]) }
```


## Support
If you have a specific problem, the best way to record this is to open an issue on GitHub. We welcome any feedback or bug reports.

## Contributing 
In the future, we will welcome pull requests to implement new features, fix bugs, add documentation or examples, or add tests to the project.
However, at present, we do not have robust CI pipelines set up to rigorously test incoming changes, and therefor will not be accepting pull requests made from outside the current team.


## Contributors
The full list of contributors is listed in `THANKS` alongside this readme. The people mentioned in this document constitute `The RoughPy Developers`.

## License
RoughPy is licensed under a BSD-3-Clause license. This was chosen specifically to match the license of NumPy.
