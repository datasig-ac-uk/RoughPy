.. _streams:

**************
Streams
**************

A stream means an object that provides the signature or log-signature over any interval.
Streams are parametrised sequential data viewed via Rough Path theory as a rough path.

To create a stream, you need **data**.
For example, we can create data using a linearly spaced time array.

.. todo::

    Is this the simplest data we can come up with? Could change this to time, temperature, paracetamol intake (3 columns).

::

    >>> times = np.linspace(0.1,1,10)
    >>> times2 = 2*times
    >>> data = np.concatenate([times.reshape(-1,1),
                               times2.reshape(-1,1),
                               (times*d2t - 2*times*dt).reshape(-1,1)], axis=1)

To create a stream, we should also provide a **context**. For more information, see :doc:`fundamentals.contexts`.

For example, we can create the following context, with Width 2, Depth 6, and Real coefficients.

.. todo::

    Comment on max depth in relation to data?

::

    >>> context = roughpy.get_context(width=2, depth=6, coeffs=roughpy.DPReal)

You can then create a **stream** using your chosen **data** and **context** with **Lie** increments.

::

    >>> stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

.. todo::

    Plot the stream, like in the electricity data example

.. todo::

    Show sig and log sig

.. todo::

    Show example with tick data, constructor

