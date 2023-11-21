.. _quickstart:

===================
RoughPy quickstart
===================

Following the NumPy (and related) convention, we import RoughPy under the alias ``rp`` as follows:

::

    import roughpy as rp

The main object(s) that you will interact with are ``Stream`` objects or the family of factory classes such as ``LieIncrementStream``. For example, we can create a ``LieIncrementStream`` using the following commands:

::

    import numpy as np
    stream = rp.LieIncrementStream.from_increments(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64), depth=2)

This will create a stream whose (hidden) underlying data are the two increments ``[0., 1., 2.]`` and ``[3., 4., 5.]``, and whose algebra elements are truncated at maximum depth 2.
To compute the log signature over an interval we use the ``log_signature`` method on the stream, for example

::

    interval = rp.RealInterval(0., 1.)
    lsig = stream.log_signature(interval)

Printing this new object ``lsig`` should give the following result

::

    { 1(2) 2(3) }

which is the first increment from the underlying data. (By default, the increments are assumed to occur at parameter values equal to their row index in the provided data.)

Similarly, the signature can be computed using the ``signature`` method on the stream object:

::

    sig = stream.signature(interval)

Notice that the ``lsig`` and ``sig`` objects have types ``Lie`` and ``FreeTensor``, respectively. They behave exactly as you would expect elements of these algebras to behave. Moreover, they will (usually) be convertible directly to a NumPy array (or TensorFlow, PyTorch, JAX tensor type in the future) of the underlying data, so you can interact with them as if they were simple arrays.

We can also construct streams by providing the raw data of Lie increments with higher order terms by specifying the width using the same constructor above.
For example, if we take width 2 and depth 2 then the elements of a Lie element will have keys ``(1, 2, [1,2])``.
So if we provide the following data, we construct a stream whose underlying Lie increments are width 2, depth 2

::

    stream = rp.LieIncrementStream.from_increments(np.array([[1, 2, 0.5], [0.2, -0.1, 0.2]], dtype=np.float64), width=2, depth=2)
    print(stream.log_signature(rp.RealInterval(0, 0.5), 2)) # returns the first increment
    # { 1(1), 2(2), 0.5([1,2]) }

    >>> print(stream.log_signature(rp.RealInterval(0, 1.5), 2)) # Campbell-Baker-Hausdorff product of both increments
    { 1.2(1) 1.9(2) 0.45([1,2]) }

.. todo::

    Add links to jupyter notebook tutorials