.. _streams:

**************
Streams
**************

^^^^^^^^^^^^^^^^^^^^^
What are streams
^^^^^^^^^^^^^^^^^^^^^

A stream means an object that provides the signature or log-signature over any interval.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do streams fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Streams are parametrised sequential data viewed via Rough Path theory as a rough path.

^^^^^^^^^^^^^^^^^^^^^^^^
How to work with streams
^^^^^^^^^^^^^^^^^^^^^^^^

To create a stream, you need **data**.
For example, we can create data using a linearly spaced time array.

::

    >>> times = np.linspace(0.1,1,10)
    >>> times2 = 2*times
    >>> data = np.concatenate([times.reshape(-1,1),
                               times2.reshape(-1,1),
                               (times*d2t - 2*times*dt).reshape(-1,1)], axis=1)

To create a stream, we should also provide a **context**. For more information, see :doc:`fundamentals.contexts`.

For example, we can create the following context, with Width 2, Depth 6, and Real coefficients. The depth specification for constructing the stream is not the depth of the data, but it is the depth of the stream. The depth of the data must be less than or equal to the depth of the stream.

::

    >>> context = roughpy.get_context(width=2, depth=6, coeffs=roughpy.DPReal)

You can then create a **stream** using your chosen **data** and **context** with **Lie** increments.

::

    >>> stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

.. todo::

    Plot the stream, like in the electricity data example. Currenly just showing a dot? Fix.

You can also compute and display the signature and log signature of any stream.

::

    >>> sig = stream.signature(interval)
    >>> print(f"{sig=!s}")
    sig={ 1() 4.5(1) 9(2) 10.125(1,1) 20.25(1,2) 20.25(2,1) 40.5(2,2) 15.1875(1,1,1) 30.375(1,1,2) 30.375(1,2,1) 60.75(1,2,2) 30.375(2,1,1) 60.75(2,1,2) 60.75(2,2,1) 121.5(2,2,2) 17.0859(1,1,1,1) 34.1719(1,1,1,2) 34.1719(1,1,2,1) 68.3438(1,1,2,2) 34.1719(1,2,1,1) 68.3438(1,2,1,2) 68.3438(1,2,2,1) 136.688(1,2,2,2) 34.1719(2,1,1,1) 68.3438(2,1,1,2) 68.3438(2,1,2,1) 136.688(2,1,2,2) 68.3438(2,2,1,1) 136.688(2,2,1,2) 136.688(2,2,2,1) 273.375(2,2,2,2) 15.3773(1,1,1,1,1) 30.7547(1,1,1,1,2) 30.7547(1,1,1,2,1) 61.5094(1,1,1,2,2) 30.7547(1,1,2,1,1) 61.5094(1,1,2,1,2) 61.5094(1,1,2,2,1) 123.019(1,1,2,2,2) 30.7547(1,2,1,1,1) 61.5094(1,2,1,1,2) 61.5094(1,2,1,2,1) 123.019(1,2,1,2,2) 61.5094(1,2,2,1,1) 123.019(1,2,2,1,2) 123.019(1,2,2,2,1) 246.038(1,2,2,2,2) 30.7547(2,1,1,1,1) 61.5094(2,1,1,1,2) 61.5094(2,1,1,2,1) 123.019(2,1,1,2,2) 61.5094(2,1,2,1,1) 123.019(2,1,2,1,2) 123.019(2,1,2,2,1) 246.038(2,1,2,2,2) 61.5094(2,2,1,1,1) 123.019(2,2,1,1,2) 123.019(2,2,1,2,1) 246.038(2,2,1,2,2) 123.019(2,2,2,1,1) 246.038(2,2,2,1,2) 246.038(2,2,2,2,1) 492.075(2,2,2,2,2) 11.533(1,1,1,1,1,1) 23.066(1,1,1,1,1,2) 23.066(1,1,1,1,2,1) 46.132(1,1,1,1,2,2) 23.066(1,1,1,2,1,1) 46.132(1,1,1,2,1,2) 46.132(1,1,1,2,2,1) 92.2641(1,1,1,2,2,2) 23.066(1,1,2,1,1,1) 46.132(1,1,2,1,1,2) 46.132(1,1,2,1,2,1) 92.2641(1,1,2,1,2,2) 46.132(1,1,2,2,1,1) 92.2641(1,1,2,2,1,2) 92.2641(1,1,2,2,2,1) 184.528(1,1,2,2,2,2) 23.066(1,2,1,1,1,1) 46.132(1,2,1,1,1,2) 46.132(1,2,1,1,2,1) 92.2641(1,2,1,1,2,2) 46.132(1,2,1,2,1,1) 92.2641(1,2,1,2,1,2) 92.2641(1,2,1,2,2,1) 184.528(1,2,1,2,2,2) 46.132(1,2,2,1,1,1) 92.2641(1,2,2,1,1,2) 92.2641(1,2,2,1,2,1) 184.528(1,2,2,1,2,2) 92.2641(1,2,2,2,1,1) 184.528(1,2,2,2,1,2) 184.528(1,2,2,2,2,1) 369.056(1,2,2,2,2,2) 23.066(2,1,1,1,1,1) 46.132(2,1,1,1,1,2) 46.132(2,1,1,1,2,1) 92.2641(2,1,1,1,2,2) 46.132(2,1,1,2,1,1) 92.2641(2,1,1,2,1,2) 92.2641(2,1,1,2,2,1) 184.528(2,1,1,2,2,2) 46.132(2,1,2,1,1,1) 92.2641(2,1,2,1,1,2) 92.2641(2,1,2,1,2,1) 184.528(2,1,2,1,2,2) 92.2641(2,1,2,2,1,1) 184.528(2,1,2,2,1,2) 184.528(2,1,2,2,2,1) 369.056(2,1,2,2,2,2) 46.132(2,2,1,1,1,1) 92.2641(2,2,1,1,1,2) 92.2641(2,2,1,1,2,1) 184.528(2,2,1,1,2,2) 92.2641(2,2,1,2,1,1) 184.528(2,2,1,2,1,2) 184.528(2,2,1,2,2,1) 369.056(2,2,1,2,2,2) 92.2641(2,2,2,1,1,1) 184.528(2,2,2,1,1,2) 184.528(2,2,2,1,2,1) 369.056(2,2,2,1,2,2) 184.528(2,2,2,2,1,1) 369.056(2,2,2,2,1,2) 369.056(2,2,2,2,2,1) 738.113(2,2,2,2,2,2) }

    >>> logsig = stream.log_signature(interval)
    >>> print(f"{logsig=!s}")
    logsig={ 4.5(1) 9(2) }

The above stream was constructed with Lie increments. You can construct streams in many ways. The six types are:
    - Tick streams
    - Piecewize abelian
    - Lie increments
    - Function streams
    - External source data (two kinds)

Tick stream and piecewise Abelian streams are shown below.

::

    stream = TickStream.from_data(data, width=2, depth=2, dtype=DPReal)
    piecewise_lie = PiecewiseAbelianStream.construct(piecewise_lie_data, width=2, depth=2, dtype=DPReal)

.. todo::
    Print the output of these

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::

.. todo::

    Include references: (create individual bib file)
        - Lyons and McLeod Signature methods in machine learning
        - Lyons and Levy Differential equations