![RoughPy Logo](branding/logo/logo_square_white.jpg)
# RoughPy
RoughPy is a package for working with streaming data as rough paths, and working with algebraic objects such as free tensors, shuffle tensors, and elements of the free Lie algebra.


## Installation
Currently, RoughPy is only available as a source distribution. It should still be installable via pip, provided you have all the dependencies set up.
Use 
```
pip install https://github.com/datasig-ac-uk/roughpy
```
The compilation process is quite long - there is a lot to build. For the full release, we will add prebuilt binaries for major platforms.
The following packages are required to build RoughPy:
 - Boost (at least version 1.81, components system threads filesystem serialization)
 - libsndfile
 - Eigen3
 - (More to be added)

Microsoft vcpkg can be used to install these dependencies - see the vcpkg.json file for the requirements.

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
Notice that the `lsig` and `sig` objects have types `Lie` and `FreeTensor`, respectively. They behave exactly as you would expect elements of these algebras to behave.
