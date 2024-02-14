.. _streams:

**************
Streams
**************

^^^^^^^^^^^^^^^^^^^^^
What are streams
^^^^^^^^^^^^^^^^^^^^^

A stream means an object that provides the signature or log-signature over any interval. For more information on streams, see :cite:t:`lyons_signature_2024` and :cite:t:`lyons_differential_2007`.

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

#. Tick streams

    ::

       # create tick stream data
       data = DATA_FORMATS[0]

       #construct stream
       >>> tick_stream = TickStream.from_data(data, width=2, depth=2, dtype=DPReal)
    .. todo::
        - Create tick stream data, tests for tick stream confusing, DATA_FORMATS = [


#. Piecewize abelian streams

    ::

        # create piecewise lie data
        piecewise_lie_data = [
                (interval,
                Lie(rng.normal(0.0, 1.0, size=(2,)), width=2, depth=2))
                for interval in piecewise_intervals
            ]

        # construct stream
        >>> piecewise_abelian_stream = PiecewiseAbelianStream.construct(piecewise_lie_data, width=2, depth=2, dtype=DPReal)

    .. todo::
        - create piecewise lie data? Similar to tests: (??)

        def piecewise_intervals(count):
            return [RealInterval(float(i), float(i + 1)) for i in range(count)]

        @pytest.fixture
        def piecewise_lie_data(piecewise_intervals, rng):
            return [
                (interval,
                Lie(rng.normal(0.0, 1.0, size=(WIDTH,)), width=WIDTH, depth=DEPTH))
                for interval in piecewise_intervals
            ]

#. Lie increment streams

    ::

        # Using lie increments data as above
        >>> lie_increment_stream = roughpy.LieIncrementStream.from_increments(data, indices=times, ctx=context)

#. Brownian streams

    ::

        >>> brownian_stream = BrownianStream.with_generator(width=2, depth=2, dtype=DPReal)

    .. todo::
        - Is this unfinished??
        - Taken from Brownian stream tests, is this ok?

#. Function streams

    ::

        >>> function_stream = ()

    .. todo::
        - Do function that returns t and 2t for function streams. Looked at test_function_path.py, confused? Returning a stream object given a function?

#. External source data (two kinds)

    ::

        >>> external_source_stream1 = ()
        >>> external_source_stream2 = ()

        >>> roughpy.ExternalDataStream.from_uri(sound_file, depth=2)

    .. todo::
        - What are the two types?

.. todo::
    - What can you print from a stream? Not the stream object. Shall we print the sig of all of these?

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::